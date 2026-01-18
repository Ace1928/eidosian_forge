import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from xformers._deprecation_warning import deprecated_function
from xformers.components import reversible as rv
from xformers.components.residual import ResidualNormStyle, get_deepnorm_coefficients
from xformers.factory.block_configs import (
from xformers.factory.block_factory import xFormerDecoderBlock, xFormerEncoderBlock
from xformers.factory.weight_init import get_weight_init_fn, xFormerWeightInit
class xFormer(torch.nn.Module):

    def __init__(self, stack_configs: Union[xFormerBlockConfig, List[xFormerBlockConfig], Dict[str, xFormerBlockConfig]], tie_embedding_weights: bool=False, weight_init: xFormerWeightInit=xFormerWeightInit.ViT):
        """
        Given a serialized configuration, generate the corresponding model.
        This is only a helper and can easily be bypassed
        """
        super().__init__()
        deprecated_function(self)
        if isinstance(stack_configs, Dict):
            stack_configs = list(stack_configs.values())
        if not isinstance(stack_configs, List):
            stack_configs = [stack_configs]
        self._verify_reversible(stack_configs)
        self._verify_deepnorm(stack_configs)
        encoders: List[torch.nn.Module] = []
        decoders: List[torch.nn.Module] = []
        self.reversible_encoder = False
        self.rev_enc_pose_encoding = None
        for config in stack_configs:
            builder = xFormerEncoderBlock.from_config if isinstance(config, xFormerEncoderConfig) else xFormerDecoderBlock.from_config
            recipient = encoders if isinstance(config, xFormerEncoderConfig) else decoders
            for i in range(config.num_layers):
                if len(recipient) > 0:
                    config.layer_position.mark_not_first()
                if config != stack_configs[-1] or i < config.num_layers - 1:
                    config.layer_position.mark_not_last()
                block = builder(config)
                if config.reversible:
                    assert isinstance(config, xFormerEncoderConfig)
                    if block.pose_encoding is not None:
                        self.rev_enc_pose_encoding = block.pose_encoding
                    self.reversible_encoder = True
                    f, g = xFormerEncoderBlock.get_reversible_layer(config)
                    recipient.append(torch.nn.ModuleList([f, g]))
                else:
                    recipient.append(block)
        assert not tie_embedding_weights or not self.reversible_encoder, 'Reversible layers and  tied embeddings is not supported for now'
        if tie_embedding_weights and encoders and encoders[0].pose_encoding and decoders and decoders[0].pose_encoding and (not config.reversible):
            logger.info('Tying encoder and decoder embeddings, as requested')
            encoders[0].pose_encoding = decoders[0].pose_encoding
        self.encoders: torch.nn.Module = rv.ReversibleSequence(torch.nn.ModuleList(encoders)) if self.reversible_encoder else torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        use_deepnorm = stack_configs[0].residual_norm_style == ResidualNormStyle.DeepNorm
        assert not use_deepnorm or not self.reversible_encoder, 'Reversible layers and deepnorm is not supported for now'
        self.init_weights(weight_init=weight_init, use_deep_norm=use_deepnorm)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(config.stack_configs, config.tie_embedding_weights, config.weight_init)

    def _verify_reversible(self, stack_configs: List[xFormerBlockConfig]):
        reversible = [c.reversible for c in filter(lambda x: x.block_type == 'encoder', stack_configs)]
        assert all(reversible) or not any(reversible), 'All layers need to have the same reversibility setting. ' + f'Currently {reversible}'

    def _verify_deepnorm(self, stack_configs: List[xFormerBlockConfig]):
        deepnorm = [c.residual_norm_style == ResidualNormStyle.DeepNorm for c in stack_configs]
        assert all(deepnorm) or not any(deepnorm), 'All layers need to have the same deepnorm setting. ' + f'Currently {deepnorm}'

    def init_weights(self, weight_init: xFormerWeightInit, use_deep_norm: bool):
        if use_deep_norm:
            encoder_coefficients, decoder_coefficients = get_deepnorm_coefficients(encoder_layers=len(self.encoders), decoder_layers=len(self.decoders))
        else:
            encoder_coefficients, decoder_coefficients = (None, None)
        encoder_gain = encoder_coefficients.beta if encoder_coefficients is not None else 1.0
        decoder_gain = decoder_coefficients.beta if decoder_coefficients is not None else 1.0
        init_fn = get_weight_init_fn(weight_init)
        for name, module in self.encoders.named_children():
            init_fn(module=module, name=name, gain=encoder_gain)
        for name, module in self.decoders.named_children():
            init_fn(module=module, name=name, gain=decoder_gain)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor]=None, encoder_input_mask: Optional[torch.Tensor]=None, decoder_input_mask: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
        if len(list(self.encoders.parameters())) > 0:
            encoders = self.encoders
            memory = src.clone()
            if isinstance(encoders, torch.nn.ModuleList):
                for encoder in encoders:
                    memory = encoder(memory, input_mask=encoder_input_mask)
            else:
                if self.rev_enc_pose_encoding:
                    memory = self.rev_enc_pose_encoding(src)
                x = torch.cat([memory, memory], dim=-1)
                if encoder_input_mask is not None:
                    if x.dim() - encoder_input_mask.dim() > 1:
                        encoder_input_mask.unsqueeze(0)
                    x += encoder_input_mask.unsqueeze(-1)
                x = encoders(x)
                memory = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)
            if not self.decoders:
                return memory
        if len(self.decoders) > 0:
            tgt = src.clone() if tgt is None else tgt
            for decoder in self.decoders:
                tgt = decoder(target=tgt, memory=memory, input_mask=decoder_input_mask)
            return tgt
        return None
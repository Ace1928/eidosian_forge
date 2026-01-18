import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
@add_start_docstrings('Bark fine acoustics model. It is a non-causal GPT-like model with `config.n_codes_total` embedding layers and\n    language modeling heads, one for each codebook.', BARK_MODEL_START_DOCSTRING.format(config='BarkFineConfig'))
class BarkFineModel(BarkPreTrainedModel):
    base_model_prefix = 'fine_acoustics'
    config_class = BarkFineConfig
    main_input_name = 'codebook_idx'

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_embeds_layers = nn.ModuleList([nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)])
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'
        self.layernorm_final = nn.LayerNorm(config.hidden_size)
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.output_vocab_size, bias=False) for _ in range(config.n_codes_given, config.n_codes_total)])
        self.gradient_checkpointing = False
        self.n_codes_total = config.n_codes_total
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embeds_layers

    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layers = new_embeddings

    def get_output_embeddings(self):
        return self.lm_heads

    def set_output_embeddings(self, new_output_embeddings):
        self.lm_heads = new_output_embeddings

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings_list = self.get_input_embeddings()
        new_embeddings_list = nn.ModuleList([self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of) for old_embeddings in old_embeddings_list])
        self.set_input_embeddings(new_embeddings_list)
        new_num_tokens = new_embeddings_list[0].weight.shape[0]
        if self.get_output_embeddings() is not None and (not self.config.tie_word_embeddings):
            old_lm_head_list = self.get_output_embeddings()
            new_lm_head_list = nn.ModuleList([self._get_resized_lm_head(old_lm_head, new_num_tokens) for old_lm_head in old_lm_head_list])
            self.set_output_embeddings(new_lm_head_list)
        return self.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds
        self.config.output_vocab_size = model_embeds[0].weight.shape[0]
        self.config.vocab_size = model_embeds[0].weight.shape[0]
        self.output_vocab_size = model_embeds[0].weight.shape[0]
        self.vocab_size = model_embeds[0].weight.shape[0]
        self.tie_weights()
        return model_embeds

    def tie_weights(self):
        """
        Tie the weights between the input embeddings list and the output embeddings list.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, 'tie_word_embeddings', True):
            self._tied_weights_keys = []
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            for i in range(self.config.n_codes_total - self.config.n_codes_given):
                self._tie_or_clone_weights(output_embeddings[i], input_embeddings[i + 1])
                self._tied_weights_keys.append(f'lm_heads.{i}.weight')
        for module in self.modules():
            if hasattr(module, '_tie_weights'):
                module._tie_weights()

    @add_start_docstrings_to_model_forward(BARK_FINE_INPUTS_DOCSTRING)
    def forward(self, codebook_idx: int, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, input_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if codebook_idx == 0:
            raise ValueError('Cannot predict 0th codebook - 0th codebook should be predicted by the coarse model')
        if input_ids is not None and input_embeds is not None:
            raise ValueError('You cannot specify both input_ids and input_embeds at the same time')
        if input_ids is None and input_embeds is None:
            raise ValueError('You have to specify either input_ids or input_embeds')
        if input_ids is not None:
            input_embeds = [input_embeds_layer(input_ids[:, :, i]).unsqueeze(-1) for i, input_embeds_layer in enumerate(self.input_embeds_layers)]
            input_embeds = torch.cat(input_embeds, dim=-1)
            input_embeds = input_embeds[:, :, :, :codebook_idx + 1].sum(dim=-1)
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else input_embeds.device
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        position_embeds = self.position_embeds_layer(position_ids)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        hidden_states = self.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(hidden_states, attention_mask=attention_mask, head_mask=head_mask[i], output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
        hidden_states = self.layernorm_final(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        logits = self.lm_heads[codebook_idx - self.config.n_codes_given](hidden_states)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented yet')
        if not return_dict:
            return tuple((v for v in [None, logits, all_hidden_states, all_self_attentions] if v is not None))
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def generate(self, coarse_output: torch.Tensor, semantic_generation_config: BarkSemanticGenerationConfig=None, coarse_generation_config: BarkCoarseGenerationConfig=None, fine_generation_config: BarkFineGenerationConfig=None, codebook_size: int=1024, history_prompt: Optional[Dict[str, torch.Tensor]]=None, **kwargs) -> torch.LongTensor:
        """
        Generates fine acoustics tokens from input coarse acoustics tokens and an additional optional `Bark` speaker
        prompt.

        Args:
            coarse_output (`torch.Tensor` of shape (batch_size, seq_len)):
                Input coarse acoustics ids, i.e the output of `BarkCoarseModel.generate`.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            coarse_generation_config (`BarkCoarseGenerationConfig`):
                Generation config indicating how to generate the coarse tokens.
            fine_generation_config (`BarkFineGenerationConfig`):
                Generation config indicating how to generate the fine tokens.
            codebook_size (`int`, *optional*, defaults to 1024):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
        Returns:
            torch.LongTensor: Output fine acoustics tokens.
        """
        if semantic_generation_config is None:
            raise ValueError('`semantic_generation_config` has to be provided')
        if coarse_generation_config is None:
            raise ValueError('`coarse_generation_config` has to be provided')
        if fine_generation_config is None:
            raise ValueError('`fine_generation_config` has to be provided')
        temperature = kwargs.get('temperature', fine_generation_config.temperature)
        max_fine_history_length = fine_generation_config.max_fine_history_length
        max_fine_input_length = fine_generation_config.max_fine_input_length
        coarse_output = coarse_output.view(coarse_output.shape[0], -1, coarse_generation_config.n_coarse_codebooks)
        coarse_output = torch.remainder(coarse_output - semantic_generation_config.semantic_vocab_size, codebook_size)
        batch_size = coarse_output.shape[0]
        if history_prompt is not None:
            x_fine_history = torch.repeat_interleave(history_prompt['fine_prompt'].T[None], batch_size, dim=0)
        else:
            x_fine_history = None
        n_coarse = coarse_generation_config.n_coarse_codebooks
        fine_input = F.pad(coarse_output, (0, fine_generation_config.n_fine_codebooks - n_coarse), 'constant', codebook_size)
        if x_fine_history is not None:
            fine_input = torch.cat([x_fine_history[:, -max_fine_history_length:, :], fine_input], dim=1)
            n_history = x_fine_history[:, -max_fine_history_length:, :].shape[1]
        else:
            n_history = 0
        n_remove_from_end = 0
        if fine_input.shape[1] < max_fine_input_length:
            n_remove_from_end = max_fine_input_length - fine_input.shape[1]
            fine_input = F.pad(fine_input, (0, 0, 0, n_remove_from_end), mode='constant', value=codebook_size)
        n_loops = (coarse_output.shape[1] - (max_fine_input_length - n_history)) / max_fine_history_length
        n_loops = int(np.ceil(n_loops))
        n_loops = max(0, n_loops) + 1
        for n_outer in range(n_loops):
            start_idx = min([n_outer * max_fine_history_length, fine_input.shape[1] - max_fine_input_length])
            start_fill_idx = min([n_history + n_outer * max_fine_history_length, fine_input.shape[1] - max_fine_history_length])
            rel_start_fill_idx = start_fill_idx - start_idx
            input_buffer = fine_input[:, start_idx:start_idx + max_fine_input_length, :]
            for n_inner in range(n_coarse, fine_generation_config.n_fine_codebooks):
                logits = self.forward(n_inner, input_buffer).logits
                if temperature is None or temperature == 1.0:
                    relevant_logits = logits[:, rel_start_fill_idx:, :codebook_size]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[:, :, :codebook_size] / temperature
                    probs = F.softmax(relevant_logits, dim=-1)[:, rel_start_fill_idx:max_fine_input_length]
                    probs = probs.reshape((-1, codebook_size))
                    codebook_preds = torch.multinomial(probs, num_samples=1).view(batch_size, -1)
                codebook_preds = codebook_preds.to(torch.int32)
                input_buffer[:, rel_start_fill_idx:, n_inner] = codebook_preds
                del logits, codebook_preds
            for n_inner in range(n_coarse, fine_generation_config.n_fine_codebooks):
                fine_input[:, start_fill_idx:start_fill_idx + (max_fine_input_length - rel_start_fill_idx), n_inner] = input_buffer[:, rel_start_fill_idx:, n_inner]
            del input_buffer
        fine_input = fine_input.transpose(1, 2)[:, :, n_history:]
        if n_remove_from_end > 0:
            fine_input = fine_input[:, :, :-n_remove_from_end]
        if fine_input.shape[-1] != coarse_output.shape[-2]:
            raise ValueError('input and output should have the same seq_len')
        return fine_input
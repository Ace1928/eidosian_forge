from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (AutoModel,
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from uform.torch_models import VisualEncoder
class VLMForCausalLM(VLMPreTrainedModel):

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.config = config
        self.text_config = AutoConfig.from_pretrained(config.text_decoder_name_or_path)
        self.text_config.vocab_size += 3
        self.text_decoder = AutoModelForCausalLM.from_config(self.text_config)
        self.image_encoder = VisualEncoder(self.config.image_encoder_hidden_size, self.config.image_encoder_patch_size, self.config.image_size, self.config.image_encoder_num_layers, self.config.image_encoder_num_heads, self.config.image_encoder_embedding_dim, self.config.image_encoder_pooling)
        for i in range(len(self.image_encoder.blocks)):
            self.image_encoder.blocks[i].ls1 = LayerScale(self.image_encoder.blocks[i].ls1.dim)
            self.image_encoder.blocks[i].ls2 = LayerScale(self.image_encoder.blocks[i].ls2.dim)
        self.image_pooler = ImageFeaturesPooler(self.config.image_encoder_hidden_size, self.text_config.hidden_size, self.config.image_pooler_num_attn_heads, self.config.image_pooler_intermediate_size, self.config.image_pooler_num_latents, self.config.initializer_range)

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def get_images_embeddings(self, images):
        features = self.image_encoder.forward_features(images)
        return self.image_pooler(features)

    def gather_continuous_embeddings(self, input_ids: torch.Tensor, word_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        start_indices = (input_ids == self.config.image_token_id).nonzero()[:, 1]
        embeddings = []
        for sample_idx, start_idx in enumerate(start_indices.tolist()):
            embeddings.append(torch.cat((word_embeddings[sample_idx, :start_idx], image_embeddings[sample_idx], word_embeddings[sample_idx, start_idx + 1:]), dim=0))
        return torch.stack(embeddings, dim=0)

    def forward(self, input_ids: torch.LongTensor=None, images: torch.Tensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[dict, Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_is or inputs_embeds')
        if inputs_embeds is None and past_key_values is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if images is not None:
                image_embeds = self.get_images_embeddings(images)
                inputs_embeds = self.gather_continuous_embeddings(input_ids, inputs_embeds, image_embeds)
        if position_ids is None:
            seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            past_key_values_length = 0
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        outputs = self.text_decoder(inputs_embeds=inputs_embeds, input_ids=input_ids if past_key_values is not None else None, attention_mask=attention_mask, labels=labels, position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, return_dict=return_dict)
        return outputs

    def prepare_inputs_for_generation(self, input_ids, images=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        if images is not None:
            model_inputs['images'] = images
        model_inputs.update({'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask, 'images': images if past_key_values is None else None})
        return model_inputs

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls._from_config(config, **kwargs)
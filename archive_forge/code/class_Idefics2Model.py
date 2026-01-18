import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...utils import (
from ..auto import AutoModel
from .configuration_idefics2 import Idefics2Config, Idefics2VisionConfig
@add_start_docstrings('Idefics2 model consisting of a SIGLIP vision encoder and Mistral language decoder', IDEFICS2_START_DOCSTRING)
class Idefics2Model(Idefics2PreTrainedModel):

    def __init__(self, config: Idefics2Config):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size
        self.vision_model = Idefics2VisionTransformer(config.vision_config)
        self.connector = Idefics2Connector(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.image_seq_len = config.perceiver_config.resampler_n_latents
        self.image_token_id = self.config.image_token_id
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                return module
            else:
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(make_inputs_require_grads)

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.text_model.resize_token_embeddings(new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def inputs_merger(self, input_ids: torch.LongTensor, inputs_embeds: Optional[torch.Tensor], image_hidden_states: Optional[torch.Tensor]):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder (and potentially the perceiver), and that hidden state is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.image_token_id
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        return new_inputs_embeds

    @add_start_docstrings_to_model_forward("\n        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to\n        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where\n        max_num_images is the maximum number of images among the batch_size samples in the batch.\n\n        Padding images are not needed beyond padding the pixel_values at the entrance of the model.\n        For efficiency, we only pass through the vision_model's forward the real images by\n        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where\n        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.\n        ", IDEFICS2_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_attention_mask: Optional[torch.BoolTensor]=None, image_hidden_states: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Idefics2BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_usable_length(seq_length)
        if inputs_embeds is not None and input_ids is None and (past_seen_tokens == 0):
            raise ValueError('When first calling the model, if input_embeds are passed, input_ids should not be None.')
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError('You cannot specify both pixel_values and image_hidden_states at the same time')
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)), dtype=torch.bool, device=pixel_values.device)
            else:
                pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
            image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask).last_hidden_state
            image_hidden_states = self.connector(image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1))
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)
        if past_seen_tokens == 0 and inputs_embeds is not None and (image_hidden_states is not None):
            inputs_embeds = self.inputs_merger(input_ids=input_ids, inputs_embeds=inputs_embeds, image_hidden_states=image_hidden_states)
        outputs = self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return tuple((v for v in [*outputs, image_hidden_states] if v is not None))
        return Idefics2BaseModelOutputWithPast(last_hidden_state=outputs.last_hidden_state, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, image_hidden_states=image_hidden_states)
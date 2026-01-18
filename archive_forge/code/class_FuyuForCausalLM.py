from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModelForCausalLM
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_fuyu import FuyuConfig
@add_start_docstrings('Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.', FUYU_START_DOCSTRING)
class FuyuForCausalLM(FuyuPreTrainedModel):

    def __init__(self, config: FuyuConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.vision_embed_tokens = nn.Linear(config.patch_size * config.patch_size * config.num_channels, config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(self, word_embeddings: torch.Tensor, continuous_embeddings: List[torch.Tensor], image_patch_input_indices: torch.Tensor) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Tensor of word embeddings.
            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
                indices in image_patch_input_indices for that batch element.
            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Tensor of indices of the image patches in the input_ids tensor.
        """
        if not word_embeddings.shape[0] == len(continuous_embeddings):
            raise ValueError(f'Batch sizes must match! Got len(continuous_embeddings)={len(continuous_embeddings)!r} and word_embeddings.shape[0]={word_embeddings.shape[0]!r}')
        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(f'Number of continuous embeddings continuous_embeddings[batch_idx].shape={continuous_embeddings[batch_idx].shape!r} does not match number of continuous token ids src_indices.shape={src_indices.shape!r} in batch element {batch_idx}.')
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    @add_start_docstrings_to_model_forward(FUYU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, image_patches: torch.Tensor=None, image_patches_indices: torch.Tensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import FuyuProcessor, FuyuForCausalLM
        >>> from PIL import Image
        >>> import requests

        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Generate a coco-style caption.\\n"

        >>> inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> generated_ids = model.generate(**model_inputs, max_new_tokens=7)
        >>> generation_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        >>> print(generation_text)
        'A bus parked on the side of a road.'
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either input_is or inputs_embeds')
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0) for patch in image_patches]
                inputs_embeds = self.gather_continuous_embeddings(word_embeddings=inputs_embeds, continuous_embeddings=patch_embeddings, image_patch_input_indices=image_patches_indices)
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, labels=labels, use_cache=use_cache, return_dict=return_dict)
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, image_patches=None, image_patches_indices=None, **kwargs):
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
        if image_patches_indices is not None:
            model_inputs['image_patches_indices'] = image_patches_indices
        model_inputs.update({'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask, 'image_patches_indices': image_patches_indices if past_key_values is None else None, 'image_patches': image_patches if past_key_values is None else None})
        return model_inputs
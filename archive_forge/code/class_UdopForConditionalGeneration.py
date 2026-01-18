import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@add_start_docstrings('The UDOP encoder-decoder Transformer with a language modeling head on top, enabling to generate text given document\n    images and an optional prompt.\n\n    This class is based on [`T5ForConditionalGeneration`], extended to deal with images and layout (2D) data.', UDOP_START_DOCSTRING)
class UdopForConditionalGeneration(UdopPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'encoder.embed_patches.proj.weight', 'encoder.embed_patches.proj.bias', 'encoder.relative_bias.biases.0.relative_attention_bias.weight', 'decoder.relative_bias.biases.0.relative_attention_bias.weight', 'lm_head.weight']

    def __init__(self, config):
        super(UdopForConditionalGeneration, self).__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = UdopPatchEmbeddings(config)
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UdopStack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(UDOP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Tensor=None, attention_mask: Tensor=None, bbox: Dict[str, Any]=None, pixel_values: Optional[Tensor]=None, visual_bbox: Dict[str, Any]=None, decoder_input_ids: Optional[Tensor]=None, decoder_attention_mask: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, encoder_outputs: Optional[Tensor]=None, past_key_values: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, decoder_inputs_embeds: Optional[Tensor]=None, decoder_head_mask: Optional[Tensor]=None, cross_attn_head_mask: Optional[Tensor]=None, use_cache=True, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[Tensor]=None) -> Tuple[Tensor, ...]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, UdopForConditionalGeneration
        >>> from datasets import load_dataset

        >>> # load model and processor
        >>> # in this case, we already have performed OCR ourselves
        >>> # so we initialize the processor with `apply_ocr=False`
        >>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        >>> model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large")

        >>> # load an example image, along with the words and coordinates
        >>> # which were extracted using an OCR engine
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> # one can use the various task prefixes (prompts) used during pre-training
        >>> # e.g. the task prefix for DocVQA is "Question answering. "
        >>> question = "Question answering. What is the date on the form?"
        >>> encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")

        >>> # autoregressive generation
        >>> predicted_ids = model.generate(**encoding)
        >>> print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
        9/30/92
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, bbox=bbox, visual_bbox=visual_bbox, pixel_values=pixel_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        encoder_attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[1]
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.config.d_model ** (-0.5)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[2:] + (encoder_outputs[0],) + encoder_outputs[2:]
            return (loss,) + output if loss is not None else output
        return Seq2SeqLMOutput(loss=loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache, 'bbox': kwargs.get('bbox', None), 'pixel_values': kwargs.get('pixel_values', None), 'visual_bbox': kwargs.get('visual_bbox', None)}

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            logger.warning('You might want to consider setting `use_cache=True` to speed up decoding')
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(f'reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched')
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(f'length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched')
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
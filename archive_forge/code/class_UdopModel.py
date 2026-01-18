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
@add_start_docstrings('The bare UDOP encoder-decoder Transformer outputting raw hidden-states without any specific head on top.', UDOP_START_DOCSTRING)
class UdopModel(UdopPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'encoder.embed_patches.proj.weight', 'encoder.embed_patches.proj.bias', 'encoder.relative_bias.biases.0.relative_attention_bias.weight', 'decoder.relative_bias.biases.0.relative_attention_bias.weight']

    def __init__(self, config):
        super(UdopModel, self).__init__(config)
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
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(UDOP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Tensor=None, attention_mask: Tensor=None, bbox: Dict[str, Any]=None, pixel_values: Optional[Tensor]=None, visual_bbox: Dict[str, Any]=None, decoder_input_ids: Optional[Tensor]=None, decoder_attention_mask: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, encoder_outputs: Optional[Tensor]=None, past_key_values: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, decoder_inputs_embeds: Optional[Tensor]=None, decoder_head_mask: Optional[Tensor]=None, cross_attn_head_mask: Optional[Tensor]=None, use_cache=True, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Tuple[Tensor, ...]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from datasets import load_dataset
        >>> import torch

        >>> # load model and processor
        >>> # in this case, we already have performed OCR ourselves
        >>> # so we initialize the processor with `apply_ocr=False`
        >>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        >>> model = AutoModel.from_pretrained("microsoft/udop-large")

        >>> # load an example image, along with the words and coordinates
        >>> # which were extracted using an OCR engine
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> inputs = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

        >>> # forward pass
        >>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1, 1024]
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values, visual_bbox=visual_bbox, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        encoder_attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[1]
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            decoder_outputs = tuple((value for idx, value in enumerate(decoder_outputs) if idx != 1))
            encoder_outputs = tuple((value for idx, value in enumerate(encoder_outputs) if idx != 1))
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)
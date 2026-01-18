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
@add_start_docstrings("The bare UDOP Model transformer outputting encoder's raw hidden-states without any specific head on top.", UDOP_START_DOCSTRING)
class UdopEncoderModel(UdopPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'encoder.embed_patches.proj.weight', 'encoder.embed_patches.proj.bias', 'encoder.relative_bias.biases.0.relative_attention_bias.weight']

    def __init__(self, config: UdopConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = UdopPatchEmbeddings(config)
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UDOP_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithAttentionMask, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Tensor=None, bbox: Dict[str, Any]=None, attention_mask: Tensor=None, pixel_values: Optional[Tensor]=None, visual_bbox: Dict[str, Any]=None, head_mask: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithAttentionMask]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, UdopEncoderModel
        >>> from huggingface_hub import hf_hub_download
        >>> from datasets import load_dataset

        >>> # load model and processor
        >>> # in this case, we already have performed OCR ourselves
        >>> # so we initialize the processor with `apply_ocr=False`
        >>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        >>> model = UdopEncoderModel.from_pretrained("microsoft/udop-large")

        >>> # load an example image, along with the words and coordinates
        >>> # which were extracted using an OCR engine
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_ids=input_ids, bbox=bbox, visual_bbox=visual_bbox, pixel_values=pixel_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return encoder_outputs
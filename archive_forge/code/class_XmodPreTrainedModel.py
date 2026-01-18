import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xmod import XmodConfig
class XmodPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XmodConfig
    base_model_prefix = 'roberta'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def set_default_language(self, language: str):
        """
        Set the default language code for the model. This is used when the language is not specified in the input.

        Args:
            language (`str`): The language code, such as `"en_XX"` or `"de_DE"`.
        """
        if language not in self.config.languages:
            raise ValueError(f'{self} does not have an adapter for {language}. Supported languages: {list(self.config.languages)}')
        self.config.default_language = language

    def freeze_embeddings_and_language_adapters(self):
        """
        Freeze the embeddings and language adapters of the model. Usually, this is applied before the model is
        fine-tuned on a downstream task.
        """
        logger.info('Freezing embeddings')
        for parameter in self.roberta.embeddings.parameters():
            parameter.requires_grad = False
        logger.info('Freezing adapters')
        for layer in self.roberta.encoder.layer:
            if layer.output.adapter_layer_norm is not None:
                for parameter in layer.output.adapter_layer_norm.parameters():
                    parameter.requires_grad = False
            for parameter in layer.output.adapter_modules.parameters():
                parameter.requires_grad = False
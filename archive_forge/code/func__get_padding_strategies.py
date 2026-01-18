from typing import Dict, List, Optional, Union
import numpy as np
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .utils import PaddingStrategy, TensorType, is_tf_tensor, is_torch_tensor, logging, to_numpy
def _get_padding_strategies(self, padding=False, max_length=None):
    """
        Find the correct padding strategy
        """
    if padding is not False:
        if padding is True:
            padding_strategy = PaddingStrategy.LONGEST
        elif not isinstance(padding, PaddingStrategy):
            padding_strategy = PaddingStrategy(padding)
        elif isinstance(padding, PaddingStrategy):
            padding_strategy = padding
    else:
        padding_strategy = PaddingStrategy.DO_NOT_PAD
    if max_length is None:
        if padding_strategy == PaddingStrategy.MAX_LENGTH:
            raise ValueError(f'When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined')
    if padding_strategy != PaddingStrategy.DO_NOT_PAD and self.padding_value is None:
        raise ValueError('Asking to pad but the feature_extractor does not have a padding value. Please select a value to use as `padding_value`. For example: `feature_extractor.padding_value = 0.0`.')
    return padding_strategy
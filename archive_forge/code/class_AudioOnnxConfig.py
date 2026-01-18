from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
class AudioOnnxConfig(OnnxConfig):
    """
    Handles audio architectures.
    """
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {'input_values': {0: 'batch_size', 1: 'sequence_length'}}
import copy
import gc
import multiprocessing as mp
import os
import traceback
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.modeling_utils import get_parameter_dtype
from transformers.utils import is_tf_available, is_torch_available
from ...onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from ...utils import (
from ...utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, MinimumVersionError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import OnnxConfig
from .constants import UNPICKABLE_ARCHS
from .model_configs import SpeechT5OnnxConfig
from .utils import (
class ValidationProcess(mp.Process):

    def __init__(self, config: OnnxConfig, reference_model: Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], onnx_model: Path, onnx_named_outputs: List[str], atol: Optional[float]=None, input_shapes: Optional[Dict]=None, device: str='cpu', model_kwargs: Optional[Dict[str, Any]]=None):
        super().__init__()
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.config = config
        self.reference_model = reference_model
        self.onnx_model = onnx_model
        self.onnx_named_outputs = onnx_named_outputs
        self.atol = atol
        self.input_shapes = input_shapes
        self.device = device
        self.model_kwargs = model_kwargs

    def run(self):
        try:
            _run_validation(config=self.config, reference_model=self.reference_model, onnx_model=self.onnx_model, onnx_named_outputs=self.onnx_named_outputs, atol=self.atol, input_shapes=self.input_shapes, device=self.device, model_kwargs=self.model_kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            return

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
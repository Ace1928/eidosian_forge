from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import TensorType, logging

        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        
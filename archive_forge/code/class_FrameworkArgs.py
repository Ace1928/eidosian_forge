import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@generate_doc_dataclass
@dataclass
class FrameworkArgs:
    opset: Optional[int] = field(default=11, metadata={'description': 'ONNX opset version to export the model with.'})
    optimization_level: Optional[int] = field(default=0, metadata={'description': 'ONNX optimization level.'})

    def __post_init__(self):
        assert self.opset <= 15, f'Unsupported OnnxRuntime opset: {self.opset}'
        assert self.optimization_level in [0, 1, 2, 99], f'Unsupported OnnxRuntime optimization level: {self.optimization_level}'
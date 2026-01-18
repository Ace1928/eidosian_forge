from abc import ABC, abstractmethod
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Optional, Set, Tuple, Union
from onnx import ModelProto, load_model
from onnxruntime.transformers.onnx_model import OnnxModel
class PreprocessorPass(ABC):

    def __init__(self):
        self._logger = LOGGER

    @abstractmethod
    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        raise NotImplementedError()
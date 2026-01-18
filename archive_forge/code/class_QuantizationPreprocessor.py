from abc import ABC, abstractmethod
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Optional, Set, Tuple, Union
from onnx import ModelProto, load_model
from onnxruntime.transformers.onnx_model import OnnxModel
class QuantizationPreprocessor:
    __slots__ = ('_passes',)

    def __init__(self):
        self._passes = []

    def from_config(self, config):
        pass

    def register_pass(self, target: PreprocessorPass):
        if target not in self._passes:
            self._passes.append(target)

    def collect(self, model_or_path: Union[str, PathLike, Path, bytes]) -> Tuple[Set[str], Set[str]]:
        global_nodes_to_quantize, global_nodes_to_exclude = (set(), set())
        graph = load_model(model_or_path.as_posix() if isinstance(model_or_path, Path) else model_or_path)
        model = OnnxModel(graph)
        for walking_pass in self._passes:
            nodes_to_quantize, nodes_to_exclude = walking_pass(graph, model)
            if nodes_to_quantize is not None:
                global_nodes_to_quantize.update(nodes_to_quantize)
            if nodes_to_exclude is not None:
                global_nodes_to_exclude.update(nodes_to_exclude)
        global_nodes_to_quantize = global_nodes_to_quantize - global_nodes_to_exclude
        return (global_nodes_to_quantize, global_nodes_to_exclude)
import sys
import types
from typing import List
import torch
class QuantizedEngine(types.ModuleType):

    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)
    engine = _QEngineProp()
    supported_engines = _SupportedQEnginesProp()
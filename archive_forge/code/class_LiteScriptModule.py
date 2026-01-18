import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
class LiteScriptModule:

    def __init__(self, cpp_module):
        self._c = cpp_module
        super().__init__()

    def __call__(self, *input):
        return self._c.forward(input)

    def find_method(self, method_name):
        return self._c.find_method(method_name)

    def forward(self, *input):
        return self._c.forward(input)

    def run_method(self, method_name, *input):
        return self._c.run_method(method_name, input)
from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
class OutOfResources(Exception):

    def __init__(self, required, limit, name):
        self.message = f'out of resource: {name}, Required: {required}, Hardware limit: {limit}. ' + 'Reducing block sizes or `num_stages` may help.'
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(self.message)

    def __reduce__(self):
        return (type(self), (self.required, self.limit, self.name))
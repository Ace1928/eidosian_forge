import types
from contextlib import contextmanager
from torch.backends import (
class PropModule(types.ModuleType):

    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)
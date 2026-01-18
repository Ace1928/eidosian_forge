import sys
import os
class DistutilsLoader(importlib.abc.Loader):

    def create_module(self, spec):
        mod.__name__ = 'distutils'
        return mod

    def exec_module(self, module):
        pass
import importlib
import logging
import os
import types
from pathlib import Path
import torch
def _import_once(self):
    if self.module is None:
        self.module = self.import_func()
        self.__dict__.update(self.module.__dict__)
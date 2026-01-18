import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
class DummyWriter:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def add_array(self, name, shape, dtype=float):
        pass

    def fill(self, a):
        pass

    def sync(self):
        pass

    def write(self, *args, **kwargs):
        pass

    def child(self, name):
        return self

    def close(self):
        pass

    def __len__(self):
        return 0
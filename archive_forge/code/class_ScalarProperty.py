from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Sequence, Union
import numpy as np
class ScalarProperty(Property):

    def __init__(self, name, dtype):
        super().__init__(name, dtype, tuple())

    def normalize_type(self, value):
        if not np.isscalar(value):
            raise TypeError('Expected scalar')
        return self.dtype(value)
import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
class BytesIOSRandomSize(BytesIO):

    def read(self, size=None):
        import random
        size = random.randint(1, size)
        return super().read(size)
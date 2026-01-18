import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
class ArrayProxy:

    def __init__(self, value):
        self.value = value

    def __array_function__(self, *args, **kwargs):
        return self.value.__array_function__(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        return self.value.__array__(*args, **kwargs)
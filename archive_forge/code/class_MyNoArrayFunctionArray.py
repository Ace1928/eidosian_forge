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
class MyNoArrayFunctionArray:

    def __init__(self, function=None):
        self.function = function
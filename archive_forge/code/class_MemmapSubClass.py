import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
class MemmapSubClass(memmap):
    pass
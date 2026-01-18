import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..spatialimages import HeaderDataError, HeaderTypeError
from ..spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage
from . import test_spm99analyze
class TestSpm2AnalyzeImage(test_spm99analyze.TestSpm99AnalyzeImage):
    image_class = Spm2AnalyzeImage
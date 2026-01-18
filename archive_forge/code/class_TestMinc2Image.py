from os.path import join as pjoin
import numpy as np
import pytest
from .. import minc2
from ..minc2 import Minc2File, Minc2Image
from ..optpkg import optional_package
from ..testing import data_path
from . import test_minc1 as tm2
class TestMinc2Image(tm2.TestMinc1Image):
    image_class = Minc2Image
    eg_images = (pjoin(data_path, 'small.mnc'),)
    module = minc2
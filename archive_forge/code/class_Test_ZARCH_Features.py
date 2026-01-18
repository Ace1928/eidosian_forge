import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
@pytest.mark.skipif(not is_linux or not is_zarch, reason='Only for Linux and IBM Z')
class Test_ZARCH_Features(AbstractTest):
    features = ['VX', 'VXE', 'VXE2']

    def load_flags(self):
        self.load_flags_auxv()
import contextlib
import functools
import inspect
import os
from platform import uname
from pathlib import Path
import shutil
import string
import sys
import warnings
from packaging.version import parse as parse_version
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import _pylab_helpers, cbook, ft2font, pyplot as plt, ticker
from .compare import comparable_formats, compare_images, make_test_filename
from .exceptions import ImageComparisonFailure
def _checked_on_freetype_version(required_freetype_version):
    import pytest
    return pytest.mark.xfail(not _check_freetype_version(required_freetype_version), reason=f"Mismatched version of freetype. Test requires '{required_freetype_version}', you have '{ft2font.__freetype_version__}'", raises=ImageComparisonFailure, strict=False)
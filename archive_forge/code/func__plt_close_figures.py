import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE, check_output
import zlib
import pytest
import numpy as np
import ase
from ase.utils import workdir, seterr
from ase.test.factories import (CalculatorInputs,
from ase.dependencies import all_dependencies
@pytest.fixture(autouse=True)
def _plt_close_figures():
    yield
    plt = sys.modules.get('matplotlib.pyplot')
    if plt is None:
        return
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.close(fignum)
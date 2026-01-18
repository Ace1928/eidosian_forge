import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def _make_bundledir(self, filename):
    """Make the main bundle directory.

        Since all MPI tasks might write to it, all tasks must wait for
        the directory to appear.
        """
    self.log('Making directory ' + filename)
    assert not os.path.isdir(filename)
    barrier()
    if self.master:
        os.mkdir(filename)
    else:
        i = 0
        while not os.path.isdir(filename):
            time.sleep(1)
            i += 1
        if i > 10:
            self.log('Waiting %d seconds for %s to appear!' % (i, filename))
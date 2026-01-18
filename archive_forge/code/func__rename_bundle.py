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
def _rename_bundle(self, oldname, newname):
    """Rename a bundle.  Used to create the .bak"""
    if self.master:
        os.rename(oldname, newname)
    else:
        while os.path.exists(oldname):
            time.sleep(1)
    barrier()
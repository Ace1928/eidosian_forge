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
def _open_log(self):
    if not (self.master or self.slavelog):
        return
    if self.master:
        lfn = os.path.join(self.filename, 'log.txt')
    else:
        lfn = os.path.join(self.filename, 'log-node%d.txt' % (world.rank,))
    self.logfile = open(lfn, 'a', 1)
    if hasattr(self, 'logdata'):
        for text in self.logdata:
            self.logfile.write(text + '\n')
        self.logfile.flush()
        del self.logdata
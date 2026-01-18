import os
import sys
import numpy as np
from ase import units
def _vprint(self, text):
    """Print output if verbose flag True."""
    if self.verbose:
        sys.stdout.write(text + os.linesep)
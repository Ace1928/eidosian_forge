import os
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp
def _mock_run(self, command=None, out=None, directory=None):
    assert False, 'Test attempted to launch a calculation'
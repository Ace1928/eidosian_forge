from typing import Dict, Any
import numpy as np
import ase
from ase.db import connect
from ase.calculators.calculator import Calculator
def _decrease_checkpoint_id(self):
    self.logfile.write('Leaving checkpoint region {0}.\n'.format(self.checkpoint_id))
    if not self.in_checkpointed_region:
        self.checkpoint_id = self.checkpoint_id[:-1]
        assert len(self.checkpoint_id) >= 1
    self.in_checkpointed_region = False
    assert self.checkpoint_id[-1] >= 1
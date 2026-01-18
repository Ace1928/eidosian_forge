import os
import os.path
from warnings import warn
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Rydberg
from ase.calculators.calculator import (Calculator, all_changes, Parameters,
def _release_force_env(self):
    """Destroys the current force-environment"""
    if self._force_env_id:
        if self._shell.isready:
            self._shell.send('DESTROY %d' % self._force_env_id)
            self._shell.expect('* READY')
        else:
            msg = 'CP2K-shell not ready, could not release force_env.'
            warn(msg, RuntimeWarning)
        self._force_env_id = None
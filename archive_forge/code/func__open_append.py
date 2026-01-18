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
def _open_append(self, atoms):
    if not os.path.exists(self.filename):
        barrier()
        self._open_write(atoms, False)
        return
    if not self.is_bundle(self.filename):
        raise IOError('Not a BundleTrajectory: ' + self.filename)
    self.state = 'read'
    metadata = self._read_metadata()
    self.metadata = metadata
    if metadata['version'] != self.version:
        raise NotImplementedError('Cannot append to a BundleTrajectory version %s (supported version is %s)' % (str(metadata['version']), str(self.version)))
    if metadata['subtype'] not in ('normal', 'split'):
        raise NotImplementedError('This version of ASE cannot append to BundleTrajectory subtype ' + metadata['subtype'])
    self.subtype = metadata['subtype']
    if metadata['backend'] == 'ulm':
        self.singleprecision = metadata['ulm.singleprecision']
    self._set_backend(metadata['backend'])
    self.nframes = self._read_nframes()
    self._open_log()
    self.log('Opening "%s" in append mode (nframes=%i)' % (self.filename, self.nframes))
    self.state = 'write'
    self.atoms = atoms
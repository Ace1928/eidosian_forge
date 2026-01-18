import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
def _generate_encfile(self):
    """Generate a topup compatible encoding file based on given directions"""
    out_file = self._get_encfilename()
    durations = self.inputs.readout_times
    if len(self.inputs.encoding_direction) != len(durations):
        if len(self.inputs.readout_times) != 1:
            raise ValueError('Readout time must be a float or match thelength of encoding directions')
        durations = durations * len(self.inputs.encoding_direction)
    lines = []
    for idx, encdir in enumerate(self.inputs.encoding_direction):
        direction = 1.0
        if encdir.endswith('-'):
            direction = -1.0
        line = [float(val[0] == encdir[0]) * direction for val in ['x', 'y', 'z']] + [durations[idx]]
        lines.append(line)
    np.savetxt(out_file, np.array(lines), fmt=b'%d %d %d %.8f')
    return out_file
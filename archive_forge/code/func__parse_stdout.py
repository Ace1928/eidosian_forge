import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
def _parse_stdout(self, stdout):
    out = []
    for string_line in stdout.split('\n'):
        if string_line.startswith('#'):
            continue
        if len(string_line) <= 1:
            continue
        line = [float(s) for s in string_line.split()]
        out.append(line)
    return np.array(out).squeeze()
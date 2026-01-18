from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def _internals(self):
    return (self._internal('xout'), self._internal('yout'), self._internal('params')[:-self.odesys.ny if self.odesys.append_iv else None])
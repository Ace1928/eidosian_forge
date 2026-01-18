import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
def _on_keypress(self, event):
    """Handle mpl keypress events"""
    if event.key is not None and 'escape' in event.key:
        self.close()
    elif event.key in ('=', '+'):
        new_idx = min(self._data_idx[3] + 1, self.n_volumes)
        self._set_volume_index(new_idx, update_slices=True)
        self._draw()
    elif event.key == '-':
        new_idx = max(self._data_idx[3] - 1, 0)
        self._set_volume_index(new_idx, update_slices=True)
        self._draw()
    elif event.key == 'ctrl+x':
        self._cross = not self._cross
        self._draw()
from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
def _metadata_pre_v3(self, char_encoding: str, sdt_control: bool) -> Dict[str, Any]:
    """Extract metadata from SPE v2 files

        Parameters
        ----------
        char_encoding
            String character encoding
        sdt_control
            If `True`, try to decode special metadata written by the
            SDT-control software.

        Returns
        -------
        dict mapping metadata names to values.

        """
    m = self._parse_header(Spec.metadata, char_encoding)
    nr = m.pop('NumROI', None)
    nr = 1 if nr < 1 else nr
    m['ROIs'] = roi_array_to_dict(m['ROIs'][:nr])
    m['chip_size'] = [m.pop(k, None) for k in ('xDimDet', 'yDimDet')]
    m['virt_chip_size'] = [m.pop(k, None) for k in ('VChipXdim', 'VChipYdim')]
    m['pre_pixels'] = [m.pop(k, None) for k in ('XPrePixels', 'YPrePixels')]
    m['post_pixels'] = [m.pop(k, None) for k in ('XPostPixels', 'YPostPixels')]
    m['comments'] = [str(c) for c in m['comments']]
    g = []
    f = m.pop('geometric', 0)
    if f & 1:
        g.append('rotate')
    if f & 2:
        g.append('reverse')
    if f & 4:
        g.append('flip')
    m['geometric'] = g
    t = m['type']
    if 1 <= t <= len(Spec.controllers):
        m['type'] = Spec.controllers[t - 1]
    else:
        m['type'] = None
    r = m['readout_mode']
    if 1 <= r <= len(Spec.readout_modes):
        m['readout_mode'] = Spec.readout_modes[r - 1]
    else:
        m['readout_mode'] = None
    for k in ('absorb_live', 'can_do_virtual_chip', 'threshold_min_live', 'threshold_max_live'):
        m[k] = bool(m[k])
    if sdt_control:
        SDTControlSpec.extract_metadata(m, char_encoding)
    return m
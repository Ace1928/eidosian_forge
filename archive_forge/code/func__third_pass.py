import glob
from os.path import join as pjoin
import numpy as np
from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file
def _third_pass(wrappers):
    """What we do when there are not unique zs in a slice set"""
    inos = [s.instance_number for s in wrappers]
    msg_fmt = 'Plausibly matching slices, but where some have the same apparent slice location, and %s; - slices are probably unsortable'
    if None in inos:
        raise DicomReadError(msg_fmt % 'some or all slices with missing InstanceNumber')
    if len(set(inos)) < len(inos):
        raise DicomReadError(msg_fmt % 'some or all slices with the same InstanceNumber')
    wrappers.sort(key=_instance_sorter)
    dw = wrappers[0]
    these_zs = [dw.slice_indicator]
    vol_list = [dw]
    out_vol_lists = [vol_list]
    for dw in wrappers[1:]:
        z = dw.slice_indicator
        if z not in these_zs:
            vol_list.append(dw)
            these_zs.append(z)
            continue
        vol_list.sort(_slice_sorter)
        vol_list = [dw]
        these_zs = [z]
        out_vol_lists.append(vol_list)
    vol_list.sort(_slice_sorter)
    return out_vol_lists
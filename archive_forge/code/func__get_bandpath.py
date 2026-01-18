import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _get_bandpath(bp):
    if bp is None:
        return []
    out = ['nwpw']
    out += _format_brillouin_zone(bp.kpts, name=bp.path)
    out += ['  zone_structure_name {}'.format(bp.path), 'end', 'task band structure']
    return out
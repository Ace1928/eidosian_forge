import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _get_kpts(atoms, **params):
    """Converts top-level 'kpts' argument to native keywords"""
    kpts = params.get('kpts')
    if kpts is None:
        return params
    nwpw = params.get('nwpw', dict())
    if 'monkhorst-pack' in nwpw or 'brillouin_zone' in nwpw:
        raise ValueError('Redundant k-points specified!')
    if isinstance(kpts, KPoints):
        nwpw['brillouin_zone'] = kpts.kpts
    elif isinstance(kpts, dict):
        if kpts.get('gamma', False) or 'size' not in kpts:
            nwpw['brillouin_zone'] = kpts2kpts(kpts, atoms).kpts
        else:
            nwpw['monkhorst-pack'] = ' '.join(map(str, kpts['size']))
    elif isinstance(kpts, np.ndarray):
        nwpw['brillouin_zone'] = kpts
    else:
        nwpw['monkhorst-pack'] = ' '.join(map(str, kpts))
    params['nwpw'] = nwpw
    return params
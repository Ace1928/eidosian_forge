from multiprocessing import Pool, cpu_count
import os.path as op
import numpy as np
import nibabel as nb
from ... import logging
from ..base import (
from .base import DipyBaseInterface
def _compute_voxel(args):
    """
    Simulate DW signal for one voxel. Uses the multi-tensor model and
    three isotropic compartments.

    Apparent diffusivity tensors are taken from [Alexander2002]_
    and [Pierpaoli1996]_.

    .. [Alexander2002] Alexander et al., Detection and modeling of non-Gaussian
      apparent diffusion coefficient profiles in human brain data, MRM
      48(2):331-340, 2002, doi: `10.1002/mrm.10209
      <https://doi.org/10.1002/mrm.10209>`_.
    .. [Pierpaoli1996] Pierpaoli et al., Diffusion tensor MR imaging
      of the human brain, Radiology 201:637-648. 1996.
    """
    from dipy.sims.voxel import multi_tensor
    ffs = args['fractions']
    gtab = args['gradients']
    signal = np.zeros_like(gtab.bvals, dtype=np.float32)
    sf_vf = np.sum(ffs)
    if sf_vf > 0.0:
        ffs = np.array(ffs) / sf_vf * 100
        snr = args['snr'] if args['snr'] > 0 else None
        try:
            signal, _ = multi_tensor(gtab, args['mevals'], S0=args['S0'], angles=args['sticks'], fractions=ffs, snr=snr)
        except Exception:
            pass
    return signal.tolist()
import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Fim(AFNICommand):
    """Program to calculate the cross-correlation of an ideal reference
    waveform with the measured FMRI time series for each voxel.

    For complete details, see the `3dfim+ Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dfim+.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> fim = afni.Fim()
    >>> fim.inputs.in_file = 'functional.nii'
    >>> fim.inputs.ideal_file= 'seed.1D'
    >>> fim.inputs.out_file = 'functional_corr.nii'
    >>> fim.inputs.out = 'Correlation'
    >>> fim.inputs.fim_thr = 0.0009
    >>> fim.cmdline
    '3dfim+ -input functional.nii -ideal_file seed.1D -fim_thr 0.000900 -out Correlation -bucket functional_corr.nii'
    >>> res = fim.run()  # doctest: +SKIP

    """
    _cmd = '3dfim+'
    input_spec = FimInputSpec
    output_spec = AFNICommandOutputSpec
import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class MRDeGibbs(MRTrix3Base):
    """
    Remove Gibbs ringing artifacts.

    This application attempts to remove Gibbs ringing artefacts from MRI images
    using the method of local subvoxel-shifts proposed by Kellner et al.

    This command is designed to run on data directly after it has been
    reconstructed by the scanner, before any interpolation of any kind has
    taken place. You should not run this command after any form of motion
    correction (e.g. not after dwipreproc). Similarly, if you intend running
    dwidenoise, you should run this command afterwards, since it has the
    potential to alter the noise structure, which would impact on dwidenoise's
    performance.

    Note that this method is designed to work on images acquired with full
    k-space coverage. Running this method on partial Fourier ('half-scan') data
    may lead to suboptimal and/or biased results, as noted in the original
    reference below. There is currently no means of dealing with this; users
    should exercise caution when using this method on partial Fourier data, and
    inspect its output for any obvious artefacts.

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/mrdegibbs.html>

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> unring = mrt.MRDeGibbs()
    >>> unring.inputs.in_file = 'dwi.mif'
    >>> unring.cmdline
    'mrdegibbs -axes 0,1 -maxW 3 -minW 1 -nshifts 20 dwi.mif dwi_unr.mif'
    >>> unring.run()                                 # doctest: +SKIP
    """
    _cmd = 'mrdegibbs'
    input_spec = MRDeGibbsInputSpec
    output_spec = MRDeGibbsOutputSpec
import os
import os.path as op
from ..base import CommandLineInputSpec, traits, TraitedSpec, File, isdefined
from .base import MRTrix3Base
class BuildConnectome(MRTrix3Base):
    """
    Generate a connectome matrix from a streamlines file and a node
    parcellation image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mat = mrt.BuildConnectome()
    >>> mat.inputs.in_file = 'tracks.tck'
    >>> mat.inputs.in_parc = 'aparc+aseg.nii'
    >>> mat.cmdline                               # doctest: +ELLIPSIS
    'tck2connectome tracks.tck aparc+aseg.nii connectome.csv'
    >>> mat.run()                                 # doctest: +SKIP
    """
    _cmd = 'tck2connectome'
    input_spec = BuildConnectomeInputSpec
    output_spec = BuildConnectomeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs
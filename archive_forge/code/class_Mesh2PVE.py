import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class Mesh2PVE(CommandLine):
    """
    Convert a mesh surface to a partial volume estimation image


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> m2p = mrt.Mesh2PVE()
    >>> m2p.inputs.in_file = 'surf1.vtk'
    >>> m2p.inputs.reference = 'dwi.mif'
    >>> m2p.inputs.in_first = 'T1.nii.gz'
    >>> m2p.cmdline                               # doctest: +ELLIPSIS
    'mesh2pve -first T1.nii.gz surf1.vtk dwi.mif mesh2volume.nii.gz'
    >>> m2p.run()                                 # doctest: +SKIP
    """
    _cmd = 'mesh2pve'
    input_spec = Mesh2PVEInputSpec
    output_spec = Mesh2PVEOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs
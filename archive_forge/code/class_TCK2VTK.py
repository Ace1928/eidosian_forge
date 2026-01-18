import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TCK2VTK(MRTrix3Base):
    """
    Convert a track file to a vtk format, cave: coordinates are in XYZ
    coordinates not reference

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> vtk = mrt.TCK2VTK()
    >>> vtk.inputs.in_file = 'tracks.tck'
    >>> vtk.inputs.reference = 'b0.nii'
    >>> vtk.cmdline                               # doctest: +ELLIPSIS
    'tck2vtk -image b0.nii tracks.tck tracks.vtk'
    >>> vtk.run()                                 # doctest: +SKIP
    """
    _cmd = 'tck2vtk'
    input_spec = TCK2VTKInputSpec
    output_spec = TCK2VTKOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs
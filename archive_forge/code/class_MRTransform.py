import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRTransform(MRTrix3Base):
    """
    Apply spatial transformations or reslice images

    Example
    -------

    >>> MRxform = MRTransform()
    >>> MRxform.inputs.in_files = 'anat_coreg.mif'
    >>> MRxform.run()                                   # doctest: +SKIP
    """
    _cmd = 'mrtransform'
    input_spec = MRTransformInputSpec
    output_spec = MRTransformOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = op.abspath(self._gen_outfilename())
        else:
            outputs['out_file'] = op.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_files[0])
        return name + '_MRTransform.mif'
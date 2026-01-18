import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class Tensor2Vector(CommandLine):
    """
    Generates a map of the major eigenvectors of the tensors in each voxel.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> tensor2vector = mrt.Tensor2Vector()
    >>> tensor2vector.inputs.in_file = 'dwi_tensor.mif'
    >>> tensor2vector.run()                             # doctest: +SKIP
    """
    _cmd = 'tensor2vector'
    input_spec = Tensor2VectorInputSpec
    output_spec = Tensor2VectorOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['vector'] = self.inputs.out_filename
        if not isdefined(outputs['vector']):
            outputs['vector'] = op.abspath(self._gen_outfilename())
        else:
            outputs['vector'] = op.abspath(outputs['vector'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_vector.mif'
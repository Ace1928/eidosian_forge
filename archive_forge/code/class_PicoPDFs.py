import os
from ...utils.filemanip import split_filename
from ..base import (
class PicoPDFs(StdOutCommandLine):
    """
    Constructs a spherical PDF in each voxel for probabilistic tractography.

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> pdf = cmon.PicoPDFs()
    >>> pdf.inputs.inputmodel = 'dt'
    >>> pdf.inputs.luts = ['lut_file']
    >>> pdf.inputs.in_file = 'voxel-order_data.Bfloat'
    >>> pdf.run()                  # doctest: +SKIP

    """
    _cmd = 'picopdfs'
    input_spec = PicoPDFsInputSpec
    output_spec = PicoPDFsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['pdfs'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_pdfs.Bdouble'
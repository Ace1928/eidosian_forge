import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class AnalyzeHeader(StdOutCommandLine):
    """
    Create or read an Analyze 7.5 header file.

    Analyze image header, provides support for the most common header fields.
    Some fields, such as patient_id, are not currently supported. The program allows
    three nonstandard options: the field image_dimension.funused1 is the image scale.
    The intensity of each pixel in the associated .img file is (image value from file) * scale.
    Also, the origin of the Talairach coordinates (midline of the anterior commisure) are encoded
    in the field data_history.originator. These changes are included for compatibility with SPM.

    All headers written with this program are big endian by default.

    Example
    -------

    >>> import nipype.interfaces.camino as cmon
    >>> hdr = cmon.AnalyzeHeader()
    >>> hdr.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> hdr.inputs.scheme_file = 'A.scheme'
    >>> hdr.inputs.data_dims = [256,256,256]
    >>> hdr.inputs.voxel_dims = [1,1,1]
    >>> hdr.run()                  # doctest: +SKIP
    """
    _cmd = 'analyzeheader'
    input_spec = AnalyzeHeaderInputSpec
    output_spec = AnalyzeHeaderOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['header'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '.hdr'
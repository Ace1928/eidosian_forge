import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class LTAConvert(CommandLine):
    """Convert different transformation formats.
    Some formats may require you to pass an image if the geometry information
    is missing form the transform file format.

    For complete details, see the `lta_convert documentation.
    <https://ftp.nmr.mgh.harvard.edu/pub/docs/html/lta_convert.help.xml.html>`_
    """
    input_spec = LTAConvertInputSpec
    output_spec = LTAConvertOutputSpec
    _cmd = 'lta_convert'

    def _format_arg(self, name, spec, value):
        if name.startswith('out_') and value is True:
            value = self._list_outputs()[name]
        return super(LTAConvert, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for name, default in (('out_lta', 'out.lta'), ('out_fsl', 'out.mat'), ('out_mni', 'out.xfm'), ('out_reg', 'out.dat'), ('out_itk', 'out.txt')):
            attr = getattr(self.inputs, name)
            if attr:
                fname = default if attr is True else attr
                outputs[name] = os.path.abspath(fname)
        return outputs
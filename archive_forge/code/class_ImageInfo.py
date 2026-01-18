import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ImageInfo(FSCommand):
    _cmd = 'mri_info'
    input_spec = ImageInfoInputSpec
    output_spec = ImageInfoOutputSpec

    def info_regexp(self, info, field, delim='\n'):
        m = re.search('%s\\s*:\\s+(.+?)%s' % (field, delim), info)
        if m:
            return m.group(1)
        else:
            return None

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        info = runtime.stdout
        outputs.info = info
        for field in ['TE', 'TR', 'TI']:
            fieldval = self.info_regexp(info, field, ', ')
            if fieldval.endswith(' msec'):
                fieldval = fieldval[:-5]
            setattr(outputs, field, fieldval)
        vox = self.info_regexp(info, 'voxel sizes')
        vox = tuple(vox.split(', '))
        outputs.vox_sizes = vox
        dim = self.info_regexp(info, 'dimensions')
        dim = tuple([int(d) for d in dim.split(' x ')])
        outputs.dimensions = dim
        outputs.orientation = self.info_regexp(info, 'Orientation')
        outputs.ph_enc_dir = self.info_regexp(info, 'PhEncDir')
        ftype, dtype = re.findall('%s\\s*:\\s+(.+?)\\n' % 'type', info)
        outputs.file_format = ftype
        outputs.data_type = dtype
        return outputs
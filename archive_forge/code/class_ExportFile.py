import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class ExportFile(SimpleInterface):
    """Export a file to an absolute path.

    This interface copies an input file to a named output file.
    This is useful to save individual files to a specific location,
    instead of more flexible interfaces like DataSink.

    Examples
    --------
    >>> from nipype.interfaces.io import ExportFile
    >>> import os.path as op
    >>> ef = ExportFile()
    >>> ef.inputs.in_file = "T1.nii.gz"
    >>> os.mkdir("output_folder")
    >>> ef.inputs.out_file = op.abspath("output_folder/sub1_out.nii.gz")
    >>> res = ef.run()
    >>> os.path.exists(res.outputs.out_file)
    True

    """
    input_spec = ExportFileInputSpec
    output_spec = ExportFileOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.clobber and op.exists(self.inputs.out_file):
            raise FileExistsError(self.inputs.out_file)
        if not op.isabs(self.inputs.out_file):
            raise ValueError('Out_file must be an absolute path.')
        if self.inputs.check_extension and split_filename(self.inputs.in_file)[2] != split_filename(self.inputs.out_file)[2]:
            raise RuntimeError('%s and %s have different extensions' % (self.inputs.in_file, self.inputs.out_file))
        shutil.copy(str(self.inputs.in_file), str(self.inputs.out_file))
        self._results['out_file'] = self.inputs.out_file
        return runtime
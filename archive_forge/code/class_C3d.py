import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
class C3d(CommandLine):
    """
    Convert3d is a command-line tool for converting 3D (or 4D) images between
    common file formats. The tool also includes a growing list of commands for
    image manipulation, such as thresholding and resampling. The tool can also
    be used to obtain information about image files. More information on
    Convert3d can be found at:
    https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md


    Example
    =======

    >>> from nipype.interfaces.c3 import C3d
    >>> c3 = C3d()
    >>> c3.inputs.in_file = "T1.nii"
    >>> c3.inputs.pix_type = "short"
    >>> c3.inputs.out_file = "T1.img"
    >>> c3.cmdline
    'c3d T1.nii -type short -o T1.img'
    >>> c3.inputs.is_4d = True
    >>> c3.inputs.in_file = "epi.nii"
    >>> c3.inputs.out_file = "epi.img"
    >>> c3.cmdline
    'c4d epi.nii -type short -o epi.img'
    """
    input_spec = C3dInputSpec
    output_spec = C3dOutputSpec
    _cmd = 'c3d'

    def __init__(self, **inputs):
        super(C3d, self).__init__(**inputs)
        self.inputs.on_trait_change(self._is_4d, 'is_4d')
        if self.inputs.is_4d:
            self._is_4d()

    def _is_4d(self):
        self._cmd = 'c4d' if self.inputs.is_4d else 'c3d'

    def _run_interface(self, runtime):
        cmd = self._cmd
        if not isdefined(self.inputs.out_file) and (not isdefined(self.inputs.out_files)):
            self._gen_outfile()
        runtime = super(C3d, self)._run_interface(runtime)
        self._cmd = cmd
        return runtime

    def _gen_outfile(self):
        if len(self.inputs.in_file) > 1 or '*' in self.inputs.in_file[0]:
            raise AttributeError('Multiple in_files found - specify either `out_file` or `out_files`.')
        _, fn, ext = split_filename(self.inputs.in_file[0])
        self.inputs.out_file = fn + '_generated' + ext
        if os.path.exists(os.path.abspath(self.inputs.out_file)):
            raise IOError('File already found - to overwrite, use `out_file`.')
        iflogger.info('Generating `out_file`.')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_files'] = os.path.abspath(self.inputs.out_file)
        if isdefined(self.inputs.out_files):
            if len(self.inputs.out_files) == 1:
                _out_files = glob(os.path.abspath(self.inputs.out_files[0]))
            else:
                _out_files = [os.path.abspath(f) for f in self.inputs.out_files if os.path.exists(os.path.abspath(f))]
            outputs['out_files'] = _out_files
        return outputs
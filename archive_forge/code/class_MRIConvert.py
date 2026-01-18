import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class MRIConvert(FSCommand):
    """use fs mri_convert to manipulate files

    .. note::
       Adds niigz as an output type option

    Examples
    --------

    >>> mc = MRIConvert()
    >>> mc.inputs.in_file = 'structural.nii'
    >>> mc.inputs.out_file = 'outfile.mgz'
    >>> mc.inputs.out_type = 'mgz'
    >>> mc.cmdline
    'mri_convert --out_type mgz --input_volume structural.nii --output_volume outfile.mgz'

    """
    _cmd = 'mri_convert'
    input_spec = MRIConvertInputSpec
    output_spec = MRIConvertOutputSpec
    filemap = dict(cor='cor', mgh='mgh', mgz='mgz', minc='mnc', afni='brik', brik='brik', bshort='bshort', spm='img', analyze='img', analyze4d='img', bfloat='bfloat', nifti1='img', nii='nii', niigz='nii.gz')

    def _format_arg(self, name, spec, value):
        if name in ['in_type', 'out_type', 'template_type']:
            if value == 'niigz':
                return spec.argstr % 'nii'
        return super(MRIConvert, self)._format_arg(name, spec, value)

    def _get_outfilename(self):
        outfile = self.inputs.out_file
        if not isdefined(outfile):
            if isdefined(self.inputs.out_type):
                suffix = '_out.' + self.filemap[self.inputs.out_type]
            else:
                suffix = '_out.nii.gz'
            outfile = fname_presuffix(self.inputs.in_file, newpath=os.getcwd(), suffix=suffix, use_ext=False)
        return os.path.abspath(outfile)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outfile = self._get_outfilename()
        if isdefined(self.inputs.split) and self.inputs.split:
            size = load(self.inputs.in_file).shape
            if len(size) == 3:
                tp = 1
            else:
                tp = size[-1]
            if outfile.endswith('.mgz'):
                stem = outfile.split('.mgz')[0]
                ext = '.mgz'
            elif outfile.endswith('.nii.gz'):
                stem = outfile.split('.nii.gz')[0]
                ext = '.nii.gz'
            else:
                stem = '.'.join(outfile.split('.')[:-1])
                ext = '.' + outfile.split('.')[-1]
            outfile = []
            for idx in range(0, tp):
                outfile.append(stem + '%04d' % idx + ext)
        if isdefined(self.inputs.out_type):
            if self.inputs.out_type in ['spm', 'analyze']:
                size = load(self.inputs.in_file).shape
                if len(size) == 3:
                    tp = 1
                else:
                    tp = size[-1]
                    raise Exception('Not taking frame manipulations into account- please warn the developers')
                outfiles = []
                outfile = self._get_outfilename()
                for i in range(tp):
                    outfiles.append(fname_presuffix(outfile, suffix='%03d' % (i + 1)))
                outfile = outfiles
        outputs['out_file'] = outfile
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._get_outfilename()
        return None
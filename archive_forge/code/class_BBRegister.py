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
class BBRegister(FSCommand):
    """Use FreeSurfer bbregister to register a volume to the Freesurfer anatomical.

    This program performs within-subject, cross-modal registration using a
    boundary-based cost function. It is required that you have an anatomical
    scan of the subject that has already been recon-all-ed using freesurfer.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import BBRegister
    >>> bbreg = BBRegister(subject_id='me', source_file='structural.nii', init='header', contrast_type='t2')
    >>> bbreg.cmdline
    'bbregister --t2 --init-header --reg structural_bbreg_me.dat --mov structural.nii --s me'

    """
    _cmd = 'bbregister'
    if LooseVersion('0.0.0') < Info.looseversion() < LooseVersion('6.0.0'):
        input_spec = BBRegisterInputSpec
    else:
        input_spec = BBRegisterInputSpec6
    output_spec = BBRegisterOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _in = self.inputs
        if isdefined(_in.out_reg_file):
            outputs['out_reg_file'] = op.abspath(_in.out_reg_file)
        elif _in.source_file:
            suffix = '_bbreg_%s.dat' % _in.subject_id
            outputs['out_reg_file'] = fname_presuffix(_in.source_file, suffix=suffix, use_ext=False)
        if isdefined(_in.registered_file):
            if isinstance(_in.registered_file, bool):
                outputs['registered_file'] = fname_presuffix(_in.source_file, suffix='_bbreg')
            else:
                outputs['registered_file'] = op.abspath(_in.registered_file)
        if isdefined(_in.out_lta_file):
            if isinstance(_in.out_lta_file, bool):
                suffix = '_bbreg_%s.lta' % _in.subject_id
                out_lta_file = fname_presuffix(_in.source_file, suffix=suffix, use_ext=False)
                outputs['out_lta_file'] = out_lta_file
            else:
                outputs['out_lta_file'] = op.abspath(_in.out_lta_file)
        if isdefined(_in.out_fsl_file):
            if isinstance(_in.out_fsl_file, bool):
                suffix = '_bbreg_%s.mat' % _in.subject_id
                out_fsl_file = fname_presuffix(_in.source_file, suffix=suffix, use_ext=False)
                outputs['out_fsl_file'] = out_fsl_file
            else:
                outputs['out_fsl_file'] = op.abspath(_in.out_fsl_file)
        if isdefined(_in.init_cost_file):
            if isinstance(_in.out_fsl_file, bool):
                outputs['init_cost_file'] = outputs['out_reg_file'] + '.initcost'
            else:
                outputs['init_cost_file'] = op.abspath(_in.init_cost_file)
        outputs['min_cost_file'] = outputs['out_reg_file'] + '.mincost'
        return outputs

    def _format_arg(self, name, spec, value):
        if name in ('registered_file', 'out_fsl_file', 'out_lta_file', 'init_cost_file') and isinstance(value, bool):
            value = self._list_outputs()[name]
        return super(BBRegister, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name == 'out_reg_file':
            return self._list_outputs()[name]
        return None
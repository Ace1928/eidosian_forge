import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Tkregister2InputSpec(FSTraitedSpec):
    target_image = File(exists=True, argstr='--targ %s', xor=['fstarg'], desc='target volume')
    fstarg = traits.Bool(False, argstr='--fstarg', xor=['target_image'], desc="use subject's T1 as reference")
    moving_image = File(exists=True, mandatory=True, argstr='--mov %s', desc='moving volume')
    fsl_in_matrix = File(exists=True, argstr='--fsl %s', desc='fsl-style registration input matrix')
    xfm = File(exists=True, argstr='--xfm %s', desc='use a matrix in MNI coordinates as initial registration')
    lta_in = File(exists=True, argstr='--lta %s', desc='use a matrix in MNI coordinates as initial registration')
    invert_lta_in = traits.Bool(requires=['lta_in'], desc='Invert input LTA before applying')
    fsl_out = traits.Either(True, File, argstr='--fslregout %s', desc='compute an FSL-compatible resgitration matrix')
    lta_out = traits.Either(True, File, argstr='--ltaout %s', desc='output registration file (LTA format)')
    invert_lta_out = traits.Bool(argstr='--ltaout-inv', requires=['lta_in'], desc='Invert input LTA before applying')
    subject_id = traits.String(argstr='--s %s', desc='freesurfer subject ID')
    noedit = traits.Bool(True, argstr='--noedit', usedefault=True, desc='do not open edit window (exit)')
    reg_file = File('register.dat', usedefault=True, mandatory=True, argstr='--reg %s', desc='freesurfer-style registration file')
    reg_header = traits.Bool(False, argstr='--regheader', desc='compute regstration from headers')
    fstal = traits.Bool(False, argstr='--fstal', xor=['target_image', 'moving_image', 'reg_file'], desc='set mov to be tal and reg to be tal xfm')
    movscale = traits.Float(argstr='--movscale %f', desc='adjust registration matrix to scale mov')
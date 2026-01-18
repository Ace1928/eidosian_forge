import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class VolumeMaskInputSpec(FSTraitedSpec):
    left_whitelabel = traits.Int(argstr='--label_left_white %d', mandatory=True, desc='Left white matter label')
    left_ribbonlabel = traits.Int(argstr='--label_left_ribbon %d', mandatory=True, desc='Left cortical ribbon label')
    right_whitelabel = traits.Int(argstr='--label_right_white %d', mandatory=True, desc='Right white matter label')
    right_ribbonlabel = traits.Int(argstr='--label_right_ribbon %d', mandatory=True, desc='Right cortical ribbon label')
    lh_pial = File(mandatory=True, exists=True, desc='Implicit input left pial surface')
    rh_pial = File(mandatory=True, exists=True, desc='Implicit input right pial surface')
    lh_white = File(mandatory=True, exists=True, desc='Implicit input left white matter surface')
    rh_white = File(mandatory=True, exists=True, desc='Implicit input right white matter surface')
    aseg = File(exists=True, xor=['in_aseg'], desc='Implicit aseg.mgz segmentation. ' + "Specify a different aseg by using the 'in_aseg' input.")
    subject_id = traits.String('subject_id', usedefault=True, position=-1, argstr='%s', mandatory=True, desc='Subject being processed')
    in_aseg = File(argstr='--aseg_name %s', exists=True, xor=['aseg'], desc='Input aseg file for VolumeMask')
    save_ribbon = traits.Bool(argstr='--save_ribbon', desc='option to save just the ribbon for the ' + 'hemispheres in the format ?h.ribbon.mgz')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the implicit input files to the ' + 'node directory.')
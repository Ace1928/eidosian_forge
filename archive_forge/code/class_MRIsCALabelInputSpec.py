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
class MRIsCALabelInputSpec(FSTraitedSpecOpenMP):
    subject_id = traits.String('subject_id', argstr='%s', position=-5, usedefault=True, mandatory=True, desc='Subject name or ID')
    hemisphere = traits.Enum('lh', 'rh', argstr='%s', position=-4, mandatory=True, desc="Hemisphere ('lh' or 'rh')")
    canonsurf = File(argstr='%s', position=-3, mandatory=True, exists=True, desc='Input canonical surface file')
    classifier = File(argstr='%s', position=-2, mandatory=True, exists=True, desc='Classifier array input file')
    smoothwm = File(mandatory=True, exists=True, desc='implicit input {hemisphere}.smoothwm')
    curv = File(mandatory=True, exists=True, desc='implicit input {hemisphere}.curv')
    sulc = File(mandatory=True, exists=True, desc='implicit input {hemisphere}.sulc')
    out_file = File(argstr='%s', position=-1, exists=False, name_source=['hemisphere'], keep_extension=True, hash_files=False, name_template='%s.aparc.annot', desc='Annotated surface output file')
    label = File(argstr='-l %s', exists=True, desc='Undocumented flag. Autorecon3 uses ../label/{hemisphere}.cortex.label as input file')
    aseg = File(argstr='-aseg %s', exists=True, desc='Undocumented flag. Autorecon3 uses ../mri/aseg.presurf.mgz as input file')
    seed = traits.Int(argstr='-seed %d', desc='')
    copy_inputs = traits.Bool(desc='Copies implicit inputs to node directory ' + 'and creates a temp subjects_directory. ' + 'Use this when running as a node')
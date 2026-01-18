import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2LabelInputSpec(FSTraitedSpec):
    hemisphere = traits.Enum('lh', 'rh', argstr='--hemi %s', mandatory=True, desc='Input hemisphere')
    subject_id = traits.String('subject_id', usedefault=True, argstr='--trgsubject %s', mandatory=True, desc='Target subject')
    sphere_reg = File(mandatory=True, exists=True, desc='Implicit input <hemisphere>.sphere.reg')
    white = File(mandatory=True, exists=True, desc='Implicit input <hemisphere>.white')
    source_sphere_reg = File(mandatory=True, exists=True, desc='Implicit input <hemisphere>.sphere.reg')
    source_white = File(mandatory=True, exists=True, desc='Implicit input <hemisphere>.white')
    source_label = File(argstr='--srclabel %s', mandatory=True, exists=True, desc='Source label')
    source_subject = traits.String(argstr='--srcsubject %s', mandatory=True, desc='Source subject name')
    out_file = File(argstr='--trglabel %s', name_source=['source_label'], name_template='%s_converted', hash_files=False, keep_extension=True, desc='Target label')
    registration_method = traits.Enum('surface', 'volume', usedefault=True, argstr='--regmethod %s', desc='Registration method')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the input files to the node ' + 'directory.')
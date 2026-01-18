import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class MS_LDAInputSpec(FSTraitedSpec):
    lda_labels = traits.List(traits.Int(), argstr='-lda %s', mandatory=True, minlen=2, maxlen=2, sep=' ', desc='pair of class labels to optimize')
    weight_file = File(argstr='-weight %s', mandatory=True, desc='filename for the LDA weights (input or output)')
    vol_synth_file = File(exists=False, argstr='-synth %s', mandatory=True, desc='filename for the synthesized output volume')
    label_file = File(exists=True, argstr='-label %s', desc='filename of the label volume')
    mask_file = File(exists=True, argstr='-mask %s', desc='filename of the brain mask volume')
    shift = traits.Int(argstr='-shift %d', desc='shift all values equal to the given value to zero')
    conform = traits.Bool(argstr='-conform', desc='Conform the input volumes (brain mask typically already conformed)')
    use_weights = traits.Bool(argstr='-W', desc='Use the weights from a previously generated weight file')
    images = InputMultiPath(File(exists=True), argstr='%s', mandatory=True, copyfile=False, desc='list of input FLASH images', position=-1)
import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class AxializeInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3daxialize', argstr='%s', position=-2, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_axialize', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    verb = traits.Bool(desc='Print out a progerss report', argstr='-verb')
    sagittal = traits.Bool(desc='Do sagittal slice order [-orient ASL]', argstr='-sagittal', xor=['coronal', 'axial'])
    coronal = traits.Bool(desc='Do coronal slice order  [-orient RSA]', argstr='-coronal', xor=['sagittal', 'axial'])
    axial = traits.Bool(desc="Do axial slice order    [-orient RAI]This is the default AFNI axial order, andis the one currently required by thevolume rendering plugin; this is alsothe default orientation output by thisprogram (hence the program's name).", argstr='-axial', xor=['coronal', 'sagittal'])
    orientation = Str(desc='new orientation code', argstr='-orient %s')
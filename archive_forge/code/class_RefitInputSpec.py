import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class RefitInputSpec(CommandLineInputSpec):
    in_file = File(desc='input file to 3drefit', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=True)
    deoblique = traits.Bool(desc='replace current transformation matrix with cardinal matrix', argstr='-deoblique')
    xorigin = Str(desc='x distance for edge voxel offset', argstr='-xorigin %s')
    yorigin = Str(desc='y distance for edge voxel offset', argstr='-yorigin %s')
    zorigin = Str(desc='z distance for edge voxel offset', argstr='-zorigin %s')
    duporigin_file = File(argstr='-duporigin %s', exists=True, desc='Copies the xorigin, yorigin, and zorigin values from the header of the given dataset')
    xdel = traits.Float(desc='new x voxel dimension in mm', argstr='-xdel %f')
    ydel = traits.Float(desc='new y voxel dimension in mm', argstr='-ydel %f')
    zdel = traits.Float(desc='new z voxel dimension in mm', argstr='-zdel %f')
    xyzscale = traits.Float(desc='Scale the size of the dataset voxels by the given factor', argstr='-xyzscale %f')
    space = traits.Enum('TLRC', 'MNI', 'ORIG', argstr='-space %s', desc='Associates the dataset with a specific template type, e.g. TLRC, MNI, ORIG')
    atrcopy = traits.Tuple(File(exists=True), traits.Str(), argstr='-atrcopy %s %s', desc="Copy AFNI header attribute from the given file into the header of the dataset(s) being modified. For more information on AFNI header attributes, see documentation file README.attributes. More than one '-atrcopy' option can be used. For AFNI advanced users only. Do NOT use -atrcopy or -atrstring with other modification options. See also -copyaux.")
    atrstring = traits.Tuple(traits.Str(), traits.Str(), argstr='-atrstring %s %s', desc='Copy the last given string into the dataset(s) being modified, giving it the attribute name given by the last string.To be safe, the last string should be in quotes.')
    atrfloat = traits.Tuple(traits.Str(), traits.Str(), argstr='-atrfloat %s %s', desc="Create or modify floating point attributes. The input values may be specified as a single string in quotes or as a 1D filename or string, example '1 0.2 0 0 -0.2 1 0 0 0 0 1 0' or flipZ.1D or '1D:1,0.2,2@0,-0.2,1,2@0,2@0,1,0'")
    atrint = traits.Tuple(traits.Str(), traits.Str(), argstr='-atrint %s %s', desc="Create or modify integer attributes. The input values may be specified as a single string in quotes or as a 1D filename or string, example '1 0 0 0 0 1 0 0 0 0 1 0' or flipZ.1D or '1D:1,0,2@0,-0,1,2@0,2@0,1,0'")
    saveatr = traits.Bool(argstr='-saveatr', desc='(default) Copy the attributes that are known to AFNI into the dset->dblk structure thereby forcing changes to known attributes to be present in the output. This option only makes sense with -atrcopy.')
    nosaveatr = traits.Bool(argstr='-nosaveatr', desc='Opposite of -saveatr')
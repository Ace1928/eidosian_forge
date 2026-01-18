import os
from ..base import (
from ...external.due import BibTeX
from .base import (
class SynthesizeInputSpec(AFNICommandInputSpec):
    cbucket = File(desc="Read the dataset output from 3dDeconvolve via the '-cbucket' option.", argstr='-cbucket %s', copyfile=False, mandatory=True)
    matrix = File(desc="Read the matrix output from 3dDeconvolve via the '-x1D' option.", argstr='-matrix %s', copyfile=False, mandatory=True)
    select = traits.List(Str(desc='selected columns to synthesize'), argstr='-select %s', desc="A list of selected columns from the matrix (and the corresponding coefficient sub-bricks from the cbucket). Valid types include 'baseline',  'polort', 'allfunc', 'allstim', 'all', Can also provide 'something' where something matches a stim_label from 3dDeconvolve, and 'digits' where digits are the numbers of the select matrix columns by numbers (starting at 0), or number ranges of the form '3..7' and '3-7'.", mandatory=True)
    out_file = File(name_template='syn', desc="output dataset prefix name (default 'syn')", argstr='-prefix %s')
    dry_run = traits.Bool(desc="Don't compute the output, just check the inputs.", argstr='-dry')
    TR = traits.Float(desc='TR to set in the output.  The default value of TR is read from the header of the matrix file.', argstr='-TR %f')
    cenfill = traits.Enum('zero', 'nbhr', 'none', argstr='-cenfill %s', desc="Determines how censored time points from the 3dDeconvolve run will be filled. Valid types are 'zero', 'nbhr' and 'none'.")
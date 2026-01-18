import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ContrastMgrInputSpec(FSLCommandInputSpec):
    tcon_file = File(exists=True, mandatory=True, argstr='%s', position=-1, desc='contrast file containing T-contrasts')
    fcon_file = File(exists=True, argstr='-f %s', desc='contrast file containing F-contrasts')
    param_estimates = InputMultiPath(File(exists=True), argstr='', copyfile=False, mandatory=True, desc='Parameter estimates for each column of the design matrix')
    corrections = File(exists=True, copyfile=False, mandatory=True, desc='statistical corrections used within FILM modelling')
    dof_file = File(exists=True, argstr='', copyfile=False, mandatory=True, desc='degrees of freedom')
    sigmasquareds = File(exists=True, argstr='', position=-2, copyfile=False, mandatory=True, desc='summary of residuals, See Woolrich, et. al., 2001')
    contrast_num = traits.Range(low=1, argstr='-cope', desc='contrast number to start labeling copes from')
    suffix = traits.Str(argstr='-suffix %s', desc='suffix to put on the end of the cope filename before the contrast number, default is nothing')
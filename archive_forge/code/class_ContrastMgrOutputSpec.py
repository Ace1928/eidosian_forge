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
class ContrastMgrOutputSpec(TraitedSpec):
    copes = OutputMultiPath(File(exists=True), desc='Contrast estimates for each contrast')
    varcopes = OutputMultiPath(File(exists=True), desc='Variance estimates for each contrast')
    zstats = OutputMultiPath(File(exists=True), desc='z-stat file for each contrast')
    tstats = OutputMultiPath(File(exists=True), desc='t-stat file for each contrast')
    fstats = OutputMultiPath(File(exists=True), desc='f-stat file for each contrast')
    zfstats = OutputMultiPath(File(exists=True), desc='z-stat file for each F contrast')
    neffs = OutputMultiPath(File(exists=True), desc='neff file ?? for each contrast')
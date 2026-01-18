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
def _get_numcons(self):
    numtcons = 0
    numfcons = 0
    if isdefined(self.inputs.tcon_file):
        fp = open(self.inputs.tcon_file, 'rt')
        for line in fp.readlines():
            if line.startswith('/NumContrasts'):
                numtcons = int(line.split()[-1])
                break
        fp.close()
    if isdefined(self.inputs.fcon_file):
        fp = open(self.inputs.fcon_file, 'rt')
        for line in fp.readlines():
            if line.startswith('/NumContrasts'):
                numfcons = int(line.split()[-1])
                break
        fp.close()
    return (numtcons, numfcons)
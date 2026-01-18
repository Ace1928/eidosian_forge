import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
class AFNICommandOutputSpec(TraitedSpec):
    out_file = File(desc='output file', exists=True)
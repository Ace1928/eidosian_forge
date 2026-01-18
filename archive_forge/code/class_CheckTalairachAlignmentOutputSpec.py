import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class CheckTalairachAlignmentOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='The input file for CheckTalairachAlignment')
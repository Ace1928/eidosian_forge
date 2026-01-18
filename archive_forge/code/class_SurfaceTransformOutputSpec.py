import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='transformed surface file')
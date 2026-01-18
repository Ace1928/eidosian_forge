import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ImageInfoInputSpec(FSTraitedSpec):
    in_file = File(exists=True, position=1, argstr='%s', desc='image to query')
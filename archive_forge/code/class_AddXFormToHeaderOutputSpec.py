import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class AddXFormToHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output volume')
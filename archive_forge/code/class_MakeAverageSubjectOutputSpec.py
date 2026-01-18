import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MakeAverageSubjectOutputSpec(TraitedSpec):
    average_subject_name = traits.Str(desc='Output registration file')
import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (

        Generate the desired figure and save the files according to
        self.inputs.output_figure_file

        
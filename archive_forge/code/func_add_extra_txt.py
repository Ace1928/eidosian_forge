from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def add_extra_txt(self, etext):
    """add additional text that will be added at the end in text format

        Parameters
        ----------
        etext : list[str]
            string with lines that are added to the text output.

        """
    self.extra_txt = '\n'.join(etext)
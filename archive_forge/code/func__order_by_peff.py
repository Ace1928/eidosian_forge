from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def _order_by_peff(self, key, criterion, reverse=True):
    self.estimator = {'min': min, 'max': max, 'mean': lambda items: sum(items) / len(items)}[criterion]
    data = []
    for sect_name, peff in self.items():
        if all((v != -1 for v in peff[key])):
            values = peff[key][:]
            if len(values) > 1:
                ref_value = values.pop(self._ref_idx)
                assert ref_value == 1.0
            data.append((sect_name, self.estimator(values)))
    data.sort(key=lambda t: t[1], reverse=reverse)
    return tuple((sect_name for sect_name, e in data))
from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _calc_site_probabilities_unordered(self, site, nn):
    bv_sum = calculate_bv_sum_unordered(site, nn, scale_factor=self.dist_scale_factor)
    prob = {}
    for specie in site.species:
        el = specie.symbol
        prob[el] = {}
        for sp, data in self.icsd_bv_data.items():
            if sp.symbol == el and sp.oxi_state != 0 and (data['std'] > 0):
                u = data['mean']
                sigma = data['std']
                prob[el][sp.oxi_state] = exp(-(bv_sum - u) ** 2 / 2 / sigma ** 2) / sigma * PRIOR_PROB[sp]
        try:
            prob[el] = {k: v / sum(prob[el].values()) for k, v in prob[el].items()}
        except ZeroDivisionError:
            prob[el] = dict.fromkeys(prob[el], 0)
    return prob
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
def _calc_site_probabilities(self, site, nn):
    el = site.specie.symbol
    bv_sum = calculate_bv_sum(site, nn, scale_factor=self.dist_scale_factor)
    prob = {}
    for sp, data in self.icsd_bv_data.items():
        if sp.symbol == el and sp.oxi_state != 0 and (data['std'] > 0):
            u = data['mean']
            sigma = data['std']
            prob[sp.oxi_state] = exp(-(bv_sum - u) ** 2 / 2 / sigma ** 2) / sigma * PRIOR_PROB[sp]
    try:
        prob = {k: v / sum(prob.values()) for k, v in prob.items()}
    except ZeroDivisionError:
        prob = dict.fromkeys(prob, 0)
    return prob
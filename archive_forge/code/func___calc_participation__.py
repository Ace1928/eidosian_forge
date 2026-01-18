from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def __calc_participation__(self):
    """ Determines, from the database, how many times each
            candidate has been used to generate new candidates. """
    participation, pairs = self.dc.get_participation_in_pairing()
    for a in self.pop:
        if a.info['confid'] in participation.keys():
            a.info['n_paired'] = participation[a.info['confid']]
        else:
            a.info['n_paired'] = 0
    self.pairs = pairs
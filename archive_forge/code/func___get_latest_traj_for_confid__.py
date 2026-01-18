import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def __get_latest_traj_for_confid__(self, confid):
    """ Method for obtaining the latest traj
            file for a given configuration.
            There can be several traj files for
            one configuration if it has undergone
            several changes (mutations, pairings, etc.)."""
    allcands = list(self.c.select(gaid=confid))
    allcands.sort(key=lambda x: x.mtime)
    return self.get_atoms(allcands[-1].id)
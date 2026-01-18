import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def __get_ids_of_all_unrelaxed_candidates__(self):
    """ Helper method used by the two above methods. """
    all_unrelaxed_ids = set([t.gaid for t in self.c.select(relaxed=0)])
    all_relaxed_ids = set([t.gaid for t in self.c.select(relaxed=1)])
    all_queued_ids = set([t.gaid for t in self.c.select(queued=1)])
    actually_unrelaxed = [gaid for gaid in all_unrelaxed_ids if gaid not in all_relaxed_ids and gaid not in all_queued_ids]
    return actually_unrelaxed
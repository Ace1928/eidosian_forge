from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
class NeighborsSetsHints:
    """
        Class used to describe neighbors sets hints.

        This allows to possibly get a lower coordination from a capped-like model polyhedron.
        """
    ALLOWED_HINTS_TYPES = ('single_cap', 'double_cap', 'triple_cap')

    def __init__(self, hints_type, options):
        """Constructor for this NeighborsSetsHints.

            Args:
                hints_type: type of hint (single, double or triple cap)
                options: options for the "hinting", e.g. the maximum csm value beyond which no additional
                    neighbors set could be found from a "cap hint".
            """
        if hints_type not in self.ALLOWED_HINTS_TYPES:
            raise ValueError(f'Type {type!r} for NeighborsSetsHints is not allowed')
        self.hints_type = hints_type
        self.options = options

    def hints(self, hints_info):
        """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
        if hints_info['csm'] > self.options['csm_max']:
            return []
        return getattr(self, f'{self.hints_type}_hints')(hints_info)

    def single_cap_hints(self, hints_info):
        """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Single cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
        cap_index_perfect = self.options['cap_index']
        nb_set = hints_info['nb_set']
        permutation = hints_info['permutation']
        nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
        cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[cap_index_perfect]
        new_site_voronoi_indices = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices.remove(cap_voronoi_index)
        return [new_site_voronoi_indices]

    def double_cap_hints(self, hints_info):
        """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Double cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
        first_cap_index_perfect = self.options['first_cap_index']
        second_cap_index_perfect = self.options['second_cap_index']
        nb_set = hints_info['nb_set']
        permutation = hints_info['permutation']
        nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
        first_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[first_cap_index_perfect]
        second_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[second_cap_index_perfect]
        new_site_voronoi_indices1 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices2 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices3 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices1.remove(first_cap_voronoi_index)
        new_site_voronoi_indices2.remove(second_cap_voronoi_index)
        new_site_voronoi_indices3.remove(first_cap_voronoi_index)
        new_site_voronoi_indices3.remove(second_cap_voronoi_index)
        return (new_site_voronoi_indices1, new_site_voronoi_indices2, new_site_voronoi_indices3)

    def triple_cap_hints(self, hints_info):
        """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Triple cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
        first_cap_index_perfect = self.options['first_cap_index']
        second_cap_index_perfect = self.options['second_cap_index']
        third_cap_index_perfect = self.options['third_cap_index']
        nb_set = hints_info['nb_set']
        permutation = hints_info['permutation']
        nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
        first_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[first_cap_index_perfect]
        second_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[second_cap_index_perfect]
        third_cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[third_cap_index_perfect]
        new_site_voronoi_indices1 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices2 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices3 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices4 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices5 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices6 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices7 = list(nb_set.site_voronoi_indices)
        new_site_voronoi_indices1.remove(first_cap_voronoi_index)
        new_site_voronoi_indices2.remove(second_cap_voronoi_index)
        new_site_voronoi_indices3.remove(third_cap_voronoi_index)
        new_site_voronoi_indices4.remove(second_cap_voronoi_index)
        new_site_voronoi_indices4.remove(third_cap_voronoi_index)
        new_site_voronoi_indices5.remove(first_cap_voronoi_index)
        new_site_voronoi_indices5.remove(third_cap_voronoi_index)
        new_site_voronoi_indices6.remove(first_cap_voronoi_index)
        new_site_voronoi_indices6.remove(second_cap_voronoi_index)
        new_site_voronoi_indices7.remove(first_cap_voronoi_index)
        new_site_voronoi_indices7.remove(second_cap_voronoi_index)
        new_site_voronoi_indices7.remove(third_cap_voronoi_index)
        return [new_site_voronoi_indices1, new_site_voronoi_indices2, new_site_voronoi_indices3, new_site_voronoi_indices4, new_site_voronoi_indices5, new_site_voronoi_indices6, new_site_voronoi_indices7]

    def as_dict(self):
        """A JSON-serializable dict representation of this NeighborsSetsHints."""
        return {'hints_type': self.hints_type, 'options': self.options}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Reconstructs the NeighborsSetsHints from its JSON-serializable dict representation."""
        return cls(hints_type=dct['hints_type'], options=dct['options'])
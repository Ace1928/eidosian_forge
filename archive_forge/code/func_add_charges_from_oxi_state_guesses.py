from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def add_charges_from_oxi_state_guesses(self, oxi_states_override: dict | None=None, target_charge: float=0, all_oxi_states: bool=False, max_sites: int | None=None) -> Composition:
    """Assign oxidation states based on guessed oxidation states.

        See `oxi_state_guesses` for an explanation of how oxidation states are
        guessed. This operation uses the set of oxidation states for each site
        that were determined to be most likely from the oxidation state guessing
        routine.

        Args:
            oxi_states_override (dict[str, list[float]]): Override an
                element's common oxidation states, e.g. {"V": [2, 3, 4, 5]}
            target_charge (float): the desired total charge on the structure.
                Default is 0 signifying charge balance.
            all_oxi_states (bool): If True, an element defaults to
                all oxidation states in pymatgen Element.icsd_oxidation_states.
                Otherwise, default is Element.common_oxidation_states. Note
                that the full oxidation state list is *very* inclusive and
                can produce nonsensical results.
            max_sites (int): If possible, will reduce Compositions to at most
                this many sites to speed up oxidation state guesses. If the
                composition cannot be reduced to this many sites a ValueError
                will be raised. Set to -1 to just reduce fully. If set to a
                number less than -1, the formula will be fully reduced but a
                ValueError will be thrown if the number of atoms in the reduced
                formula is greater than abs(max_sites).

        Returns:
            Composition, where the elements are assigned oxidation states based
            on the results form guessing oxidation states. If no oxidation state
            is possible, returns a Composition where all oxidation states are 0.
        """
    _, oxidation_states = self._get_oxi_state_guesses(all_oxi_states, max_sites, oxi_states_override, target_charge)
    if not oxidation_states:
        return Composition({Species(e, 0): f for e, f in self.items()})
    species = []
    for el, charges in oxidation_states[0].items():
        species.extend([Species(el, c) for c in charges])
    return Composition(collections.Counter(species))
from __future__ import annotations
import json
import os
import warnings
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
class GrandPotentialInterfacialReactivity(InterfacialReactivity):
    """
    Extends upon InterfacialReactivity to allow for modelling possible reactions
    at the interface between two solids in the presence of an open element. The
    thermodynamics of the open system are provided by the user via the
    GrandPotentialPhaseDiagram class.
    """

    def __init__(self, c1: Composition, c2: Composition, grand_pd: GrandPotentialPhaseDiagram, pd_non_grand: PhaseDiagram, include_no_mixing_energy: bool=False, norm: bool=True, use_hull_energy: bool=True):
        """
        Args:
            c1: Reactant 1 composition
            c2: Reactant 2 composition
            grand_pd: Grand potential phase diagram object built from all elements in
                composition c1 and c2.
            include_no_mixing_energy: No_mixing_energy for a reactant is the
                opposite number of its energy above grand potential convex hull. In
                cases where reactions involve elements reservoir, this param
                determines whether no_mixing_energy of reactants will be included
                in the final reaction energy calculation. By definition, if pd is
                not a GrandPotentialPhaseDiagram object, this param is False.
            pd_non_grand: PhaseDiagram object but not
                GrandPotentialPhaseDiagram object built from elements in c1 and c2.
            norm: Whether or not the total number of atoms in composition
                of reactant will be normalized to 1.
            use_hull_energy: Whether or not use the convex hull energy for
                a given composition for reaction energy calculation. If false,
                the energy of ground state structure will be used instead.
                Note that in case when ground state can not be found for a
                composition, convex hull energy will be used associated with a
                warning message.
        """
        if not isinstance(grand_pd, GrandPotentialPhaseDiagram):
            raise ValueError('Please use the InterfacialReactivity class if using a regular phase diagram!')
        if not isinstance(pd_non_grand, PhaseDiagram):
            raise ValueError('Please provide non-grand phase diagram to compute no_mixing_energy!')
        super().__init__(c1=c1, c2=c2, pd=grand_pd, norm=norm, use_hull_energy=use_hull_energy, bypass_grand_warning=True)
        self.pd_non_grand = pd_non_grand
        self.grand = True
        self.comp1 = Composition({k: v for k, v in c1.items() if k not in grand_pd.chempots})
        self.comp2 = Composition({k: v for k, v in c2.items() if k not in grand_pd.chempots})
        if self.norm:
            self.factor1 = self.comp1.num_atoms / c1.num_atoms
            self.factor2 = self.comp2.num_atoms / c2.num_atoms
            self.comp1 = self.comp1.fractional_composition
            self.comp2 = self.comp2.fractional_composition
        if include_no_mixing_energy:
            self.e1 = self._get_grand_potential(self.c1)
            self.e2 = self._get_grand_potential(self.c2)
        else:
            self.e1 = self.pd.get_hull_energy(self.comp1)
            self.e2 = self.pd.get_hull_energy(self.comp2)

    def get_no_mixing_energy(self):
        """
        Generates the opposite number of energy above grand potential
        convex hull for both reactants.

        Returns:
            [(reactant1, no_mixing_energy1),(reactant2,no_mixing_energy2)].
        """
        energy1 = self.pd.get_hull_energy(self.comp1) - self._get_grand_potential(self.c1)
        energy2 = self.pd.get_hull_energy(self.comp2) - self._get_grand_potential(self.c2)
        unit = 'eV/f.u.'
        if self.norm:
            unit = 'eV/atom'
        return [(f'{self.c1_original.reduced_formula} ({unit})', energy1), (f'{self.c2_original.reduced_formula} ({unit})', energy2)]

    def _get_reactants(self, x: float) -> list[Composition]:
        """Returns a list of relevant reactant compositions given an x coordinate."""
        reactants = super()._get_reactants(x)
        reactants += [Composition(entry.symbol) for entry in self.pd.chempots]
        return reactants

    def _get_grand_potential(self, composition: Composition) -> float:
        """
        Computes the grand potential Phi at a given composition and
        chemical potential(s).

        Args:
            composition: Composition object.

        Returns:
            Grand potential at a given composition at chemical potential(s).
        """
        if self.use_hull_energy:
            grand_potential = self.pd_non_grand.get_hull_energy(composition)
        else:
            grand_potential = self._get_entry_energy(self.pd_non_grand, composition)
        grand_potential -= sum((composition[e] * mu for e, mu in self.pd.chempots.items()))
        if self.norm:
            grand_potential /= sum((composition[el] for el in composition if el not in self.pd.chempots))
        return grand_potential
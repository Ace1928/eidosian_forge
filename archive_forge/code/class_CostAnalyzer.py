from the a CostDB instance, for example a CSV file via CostDBCSV.
from __future__ import annotations
import abc
import csv
import itertools
import os
from collections import defaultdict
import scipy.constants as const
from monty.design_patterns import singleton
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.util.provenance import is_valid_bibtex
class CostAnalyzer:
    """Given a CostDB, figures out the minimum cost solutions via convex hull."""

    def __init__(self, costdb):
        """
        Args:
            costdb (): Cost database.
        """
        self.costdb = costdb

    def get_lowest_decomposition(self, composition):
        """
        Get the decomposition leading to lowest cost.

        Args:
            composition:
                Composition as a pymatgen.core.structure.Composition

        Returns:
            Decomposition as a dict of {Entry: amount}
        """
        entries_list = []
        elements = [e.symbol for e in composition.elements]
        for idx in range(len(elements)):
            for combi in itertools.combinations(elements, idx + 1):
                chemsys = [Element(e) for e in combi]
                x = self.costdb.get_entries(chemsys)
                entries_list.extend(x)
        try:
            pd = PhaseDiagram(entries_list)
            return pd.get_decomposition(composition)
        except IndexError:
            raise ValueError('Error during PD building; most likely, cost data does not exist!')

    def get_cost_per_mol(self, comp):
        """
        Get best estimate of minimum cost/mol based on known data.

        Args:
            comp:
                Composition as a pymatgen.core.structure.Composition

        Returns:
            float of cost/mol
        """
        comp = comp if isinstance(comp, Composition) else Composition(comp)
        decomp = self.get_lowest_decomposition(comp)
        return sum((k.energy_per_atom * v * comp.num_atoms for k, v in decomp.items()))

    def get_cost_per_kg(self, comp):
        """
        Get best estimate of minimum cost/kg based on known data.

        Args:
            comp:
                Composition as a pymatgen.core.structure.Composition

        Returns:
            float of cost/kg
        """
        comp = comp if isinstance(comp, Composition) else Composition(comp)
        return self.get_cost_per_mol(comp) / (comp.weight.to('kg') * const.N_A)
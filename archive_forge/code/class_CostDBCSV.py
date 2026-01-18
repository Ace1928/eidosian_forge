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
class CostDBCSV(CostDB):
    """
    Read a CSV file to get costs
    Format is formula,cost_per_kg,name,BibTeX.
    """

    def __init__(self, filename):
        """
        Args:
            filename (str): Filename of cost database.
        """
        self._chemsys_entries = defaultdict(list)
        filename = os.path.join(os.path.dirname(__file__), filename)
        with open(filename) as file:
            reader = csv.reader(file, quotechar='|')
            for row in reader:
                comp = Composition(row[0])
                cost_per_mol = float(row[1]) * comp.weight.to('kg') * const.N_A
                pde = CostEntry(comp.formula, cost_per_mol, row[2], row[3])
                chemsys = '-'.join(sorted((el.symbol for el in pde.elements)))
                self._chemsys_entries[chemsys].append(pde)

    def get_entries(self, chemsys):
        """
        For a given chemical system, return an array of CostEntries.

        Args:
            chemsys:
                array of Elements defining the chemical system.

        Returns:
            array of CostEntries
        """
        chemsys = '-'.join(sorted((el.symbol for el in chemsys)))
        return self._chemsys_entries[chemsys]
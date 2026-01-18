from a list of entries within a chemical system containing 2 or more elements. The
from __future__ import annotations
import json
import os
import warnings
from functools import lru_cache
from itertools import groupby
from typing import TYPE_CHECKING
import numpy as np
import plotly.express as px
from monty.json import MSONable
from plotly.graph_objects import Figure, Mesh3d, Scatter, Scatter3d
from scipy.spatial import ConvexHull, HalfspaceIntersection
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition, Element
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.string import htmlify
def _get_domains(self) -> dict[str, np.ndarray]:
    """Returns a dictionary of domains as {formula: np.ndarray}."""
    hyperplanes = self._hyperplanes
    border_hyperplanes = self._border_hyperplanes
    entries = self._hyperplane_entries
    hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
    interior_point = np.min(self.lims, axis=1) + 0.1
    hs_int = HalfspaceIntersection(hs_hyperplanes, interior_point)
    domains = {entry.reduced_formula: [] for entry in entries}
    for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
        for v in facet:
            if v < len(entries):
                this_entry = entries[v]
                formula = this_entry.reduced_formula
                domains[formula].append(intersection)
    return {k: np.array(v) for k, v in domains.items() if v}
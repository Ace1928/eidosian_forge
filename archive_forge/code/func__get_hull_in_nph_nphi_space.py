from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
def _get_hull_in_nph_nphi_space(self, entries) -> tuple[list[PourbaixEntry], list[Simplex]]:
    """
        Generates convex hull of Pourbaix diagram entries in composition,
        npH, and nphi space. This enables filtering of multi-entries
        such that only compositionally stable combinations of entries
        are included.

        Args:
            entries ([PourbaixEntry]): list of PourbaixEntries to construct
                the convex hull

        Returns:
            tuple[list[PourbaixEntry], list[Simplex]]: PourbaixEntry list and stable
                facets corresponding to that list
        """
    ion_entries = [entry for entry in entries if entry.phase_type == 'Ion']
    solid_entries = [entry for entry in entries if entry.phase_type == 'Solid']
    logger.debug('Pre-filtering solids by min energy at each composition')
    sorted_entries = sorted(solid_entries, key=lambda x: (x.composition.reduced_composition, x.entry.energy_per_atom))
    grouped_by_composition = itertools.groupby(sorted_entries, key=lambda x: x.composition.reduced_composition)
    min_entries = [next(iter(grouped_entries)) for comp, grouped_entries in grouped_by_composition]
    min_entries += ion_entries
    logger.debug('Constructing nph-nphi-composition points for qhull')
    vecs = self._convert_entries_to_points(min_entries)
    maxes = np.max(vecs[:, :3], axis=0)
    extra_point = np.concatenate([maxes, np.ones(self.dim) / self.dim], axis=0)
    pad = 1000
    extra_point[2] += pad
    points = np.concatenate([vecs, np.array([extra_point])], axis=0)
    logger.debug('Constructing convex hull in nph-nphi-composition space')
    hull = ConvexHull(points, qhull_options='QJ i')
    facets = [facet for facet in hull.simplices if len(points) - 1 not in facet]
    if self.dim > 1:
        logger.debug('Filtering facets by Pourbaix composition')
        valid_facets = []
        for facet in facets:
            comps = vecs[facet][:, 3:]
            full_comps = np.concatenate([comps, 1 - np.sum(comps, axis=1).reshape(len(comps), 1)], axis=1)
            if np.linalg.matrix_rank(full_comps) > self.dim:
                valid_facets.append(facet)
    else:
        valid_facets = facets
    return (min_entries, valid_facets)
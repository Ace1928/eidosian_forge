from __future__ import annotations
from collections.abc import Generator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING
def _determine_grid_dimensions(self, facet_spec: FacetSpec, pair_spec: PairSpec) -> None:
    """Parse faceting and pairing information to define figure structure."""
    self.grid_dimensions: dict[str, list] = {}
    for dim, axis in zip(['col', 'row'], ['x', 'y']):
        facet_vars = facet_spec.get('variables', {})
        if dim in facet_vars:
            self.grid_dimensions[dim] = facet_spec['structure'][dim]
        elif axis in pair_spec.get('structure', {}):
            self.grid_dimensions[dim] = [None for _ in pair_spec.get('structure', {})[axis]]
        else:
            self.grid_dimensions[dim] = [None]
        self.subplot_spec[f'n{dim}s'] = len(self.grid_dimensions[dim])
    if not pair_spec.get('cross', True):
        self.subplot_spec['nrows'] = 1
    self.n_subplots = self.subplot_spec['ncols'] * self.subplot_spec['nrows']
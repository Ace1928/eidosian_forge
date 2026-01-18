import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
class Selections(traitlets.HasTraits):
    """
    Traitlet class storing a JupyterChart's selections
    """

    def __init__(self, trait_values):
        super().__init__()
        for key, value in trait_values.items():
            if isinstance(value, IndexSelection):
                traitlet_type = traitlets.Instance(IndexSelection)
            elif isinstance(value, PointSelection):
                traitlet_type = traitlets.Instance(PointSelection)
            elif isinstance(value, IntervalSelection):
                traitlet_type = traitlets.Instance(IntervalSelection)
            else:
                raise ValueError(f'Unexpected selection type: {type(value)}')
            self.add_traits(**{key: traitlet_type})
            setattr(self, key, value)
            self.observe(self._make_read_only, names=key)

    def __repr__(self):
        return f'Selections({self.trait_values()})'

    def _make_read_only(self, change):
        """
        Work around to make traits read-only, but still allow us to change
        them internally
        """
        if change['name'] in self.traits() and change['old'] != change['new']:
            self._set_value(change['name'], change['old'])
        raise ValueError(f'Selections may not be set from Python.\nAttempted to set select: {change['name']}')

    def _set_value(self, key, value):
        self.unobserve(self._make_read_only, names=key)
        setattr(self, key, value)
        self.observe(self._make_read_only, names=key)
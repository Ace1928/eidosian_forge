import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
@traitlets.observe('_vl_selections')
def _on_change_selections(self, change):
    """
        Internal callback function that updates the JupyterChart's public
        selections traitlet in response to changes that the JavaScript logic
        makes to the internal _selections traitlet.
        """
    for selection_name, selection_dict in change.new.items():
        value = selection_dict['value']
        store = selection_dict['store']
        selection_type = self._selection_types[selection_name]
        if selection_type == 'index':
            self.selections._set_value(selection_name, IndexSelection.from_vega(selection_name, signal=value, store=store))
        elif selection_type == 'point':
            self.selections._set_value(selection_name, PointSelection.from_vega(selection_name, signal=value, store=store))
        elif selection_type == 'interval':
            self.selections._set_value(selection_name, IntervalSelection.from_vega(selection_name, signal=value, store=store))
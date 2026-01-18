import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
class LegendControl(Control):
    """LegendControl class, with Control as parent class.

    A control which contains a legend.

    .. deprecated :: 0.17.0
       The constructor argument 'name' is deprecated, use the 'title' argument instead.


    Attributes
    ----------
    title: str, default 'Legend'
        The title of the legend.
    legend: dict, default 'Legend'
        A dictionary containing names as keys and CSS colors as values.
    """
    _view_name = Unicode('LeafletLegendControlView').tag(sync=True)
    _model_name = Unicode('LeafletLegendControlModel').tag(sync=True)
    title = Unicode('Legend').tag(sync=True)
    legend = Dict(default_value={'value 1': '#AAF', 'value 2': '#55A', 'value 3': '#005'}).tag(sync=True)

    def __init__(self, legend, *args, **kwargs):
        kwargs['legend'] = legend
        if 'name' in kwargs:
            warnings.warn('the name argument is deprecated, use title instead', DeprecationWarning)
            kwargs.setdefault('title', kwargs['name'])
            del kwargs['name']
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        """The title of the legend.

        .. deprecated :: 0.17.0
           Use title attribute instead.
        """
        warnings.warn('.name is deprecated, use .title instead', DeprecationWarning)
        return self.title

    @name.setter
    def name(self, title):
        warnings.warn('.name is deprecated, use .title instead', DeprecationWarning)
        self.title = title

    @property
    def legends(self):
        """The legend information.

        .. deprecated :: 0.17.0
           Use legend attribute instead.
        """
        warnings.warn('.legends is deprecated, use .legend instead', DeprecationWarning)
        return self.legend

    @legends.setter
    def legends(self, legends):
        warnings.warn('.legends is deprecated, use .legend instead', DeprecationWarning)
        self.legend = legends

    @property
    def positioning(self):
        """The position information.

        .. deprecated :: 0.17.0
           Use position attribute instead.
        """
        warnings.warn('.positioning is deprecated, use .position instead', DeprecationWarning)
        return self.position

    @positioning.setter
    def positioning(self, position):
        warnings.warn('.positioning is deprecated, use .position instead', DeprecationWarning)
        self.position = position

    @property
    def positionning(self):
        """The position information.

        .. deprecated :: 0.17.0
           Use position attribute instead.
        """
        warnings.warn('.positionning is deprecated, use .position instead', DeprecationWarning)
        return self.position

    @positionning.setter
    def positionning(self, position):
        warnings.warn('.positionning is deprecated, use .position instead', DeprecationWarning)
        self.position = position

    def add_legend_element(self, key, value):
        """Add a new legend element.

        Parameters
        ----------
        key: str
            The key for the legend element.
        value: CSS Color
            The value for the legend element.
        """
        self.legend[key] = value
        self.send_state()

    def remove_legend_element(self, key):
        """Remove a legend element.

        Parameters
        ----------
        key: str
            The element to remove.
        """
        del self.legend[key]
        self.send_state()
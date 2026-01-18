import numpy as np
import param
from ..core import Dataset
from ..core.data import MultiInterface
from ..core.dimension import Dimension
from .geom import Geometry
from .selection import SelectionPolyExpr
class BaseShape(Path):
    """
    A BaseShape is a Path that can be succinctly expressed by a small
    number of parameters instead of a full path specification. For
    instance, a circle may be expressed by the center position and
    radius instead of an explicit list of path coordinates.
    """
    __abstract = True

    def __new__(cls, *args, **kwargs):
        return super(Dataset, cls).__new__(cls)

    def __init__(self, **params):
        super().__init__([], **params)
        self.interface = MultiInterface

    def clone(self, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.
        """
        link = overrides.pop('link', True)
        settings = dict(self.param.values(), **overrides)
        if 'id' not in settings:
            settings['id'] = self.id
        if not args and link:
            settings['plot_id'] = self._plot_id
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        return self.__class__(*(settings[n] for n in pos_args), **{k: v for k, v in settings.items() if k not in pos_args})
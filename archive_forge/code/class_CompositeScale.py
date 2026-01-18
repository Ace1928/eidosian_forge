from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Instance, Required
from .transforms import Transform
class CompositeScale(Scale):
    """ Represent a composition of two scales, which useful for defining
    sub-coordinate systems.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    source_scale = Required(Instance(Scale), help='\n    The source scale.\n    ')
    target_scale = Required(Instance(Scale), help='\n    The target scale.\n    ')
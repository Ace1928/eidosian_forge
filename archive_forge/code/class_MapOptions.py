from __future__ import annotations
import logging # isort:skip
from ..core.enums import MapType
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..models.ranges import Range1d
from .plots import Plot
@abstract
class MapOptions(Model):
    """ Abstract base class for map options' models.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    lat = Required(Float, help='\n    The latitude where the map should be centered.\n    ')
    lng = Required(Float, help='\n    The longitude where the map should be centered.\n    ')
    zoom = Int(12, help='\n    The initial zoom level to use when displaying the map.\n    ')
from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import Instance, InstanceDefault, Override
from ..renderers.renderer import CompositeRenderer
from ..sources import ColumnDataSource, DataSource
@abstract
class DataAnnotation(Annotation):
    """ Base class for annotations that utilize a data source.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    source = Instance(DataSource, default=InstanceDefault(ColumnDataSource), help='\n    Local data source to use when rendering annotations on the plot.\n    ')
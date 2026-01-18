from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import Instance, InstanceDefault, Override
from ..renderers.renderer import CompositeRenderer
from ..sources import ColumnDataSource, DataSource
 Base class for annotations that utilize a data source.

    
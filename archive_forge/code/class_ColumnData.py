from __future__ import annotations
import logging # isort:skip
from collections.abc import (
from typing import TYPE_CHECKING, Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .descriptors import ColumnDataPropertyDescriptor
from .enum import Enum
from .numeric import Int
from .singletons import Intrinsic, Undefined
from .wrappers import (
class ColumnData(Dict):
    """ Accept a Python dictionary suitable as the ``data`` attribute of a
    :class:`~bokeh.models.sources.ColumnDataSource`.

    This class is a specialization of ``Dict`` that handles efficiently
    encoding columns that are NumPy arrays.

    """

    def make_descriptors(self, base_name):
        """ Return a list of ``ColumnDataPropertyDescriptor`` instances to
        install on a class, in order to delegate attribute access to this
        property.

        Args:
            base_name (str) : the name of the property these descriptors are for

        Returns:
            list[ColumnDataPropertyDescriptor]

        The descriptors returned are collected by the ``MetaHasProps``
        metaclass and added to ``HasProps`` subclasses during class creation.
        """
        return [ColumnDataPropertyDescriptor(base_name, self)]

    def _hinted_value(self, value: Any, hint: DocumentPatchedEvent | None) -> Any:
        from ...document.events import ColumnDataChangedEvent, ColumnsStreamedEvent
        if isinstance(hint, ColumnDataChangedEvent):
            return {col: hint.model.data[col] for col in hint.cols}
        if isinstance(hint, ColumnsStreamedEvent):
            return hint.data
        return value

    def wrap(self, value):
        """ Some property types need to wrap their values in special containers, etc.

        """
        if isinstance(value, dict):
            if isinstance(value, PropertyValueColumnData):
                return value
            else:
                return PropertyValueColumnData(value)
        else:
            return value
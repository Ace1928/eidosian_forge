from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
class ColumnDataPropertyDescriptor(PropertyDescriptor):
    """ A ``PropertyDescriptor`` specialized to handling ``ColumnData`` properties.

    """

    def __set__(self, obj, value, *, setter=None):
        """ Implement the setter for the Python `descriptor protocol`_.

        This method first separately extracts and removes any ``units`` field
        in the JSON, and sets the associated units property directly. The
        remaining value is then passed to the superclass ``__set__`` to
        be handled.

        .. note::
            An optional argument ``setter`` has been added to the standard
            setter arguments. When needed, this value should be provided by
            explicitly invoking ``__set__``. See below for more information.

        Args:
            obj (HasProps) :
                The instance to set a new property value on

            value (obj) :
                The new value to set the property to

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                In the context of a Bokeh server application, incoming updates
                to properties will be annotated with the session that is
                doing the updating. This value is propagated through any
                subsequent change notifications that the update triggers.
                The session can compare the event setter to itself, and
                suppress any updates that originate from itself.

        Returns:
            None

        """
        if not hasattr(obj, '_property_values'):
            class_name = obj.__class__.__name__
            raise RuntimeError(f'Cannot set a property value {self.name!r} on a {class_name} instance before HasProps.__init__')
        if self.property.readonly and obj._initialized:
            class_name = obj.__class__.__name__
            raise RuntimeError(f'{class_name}.{self.name} is a readonly property')
        if isinstance(value, PropertyValueColumnData):
            raise ValueError(_CDS_SET_FROM_CDS_ERROR)
        from ...document.events import ColumnDataChangedEvent
        hint = ColumnDataChangedEvent(obj.document, obj, 'data', setter=setter) if obj.document else None
        value = self.property.prepare_value(obj, self.name, value)
        old = self._get(obj)
        self._set(obj, old, value, hint=hint, setter=setter)
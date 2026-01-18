from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
class DataSpecPropertyDescriptor(PropertyDescriptor):
    """ A ``PropertyDescriptor`` for Bokeh |DataSpec| properties that serialize to
    field/value dictionaries.

    """

    def get_value(self, obj: HasProps) -> Any:
        """

        """
        return self.property.to_serializable(obj, self.name, getattr(obj, self.name))

    def set_from_json(self, obj: HasProps, value: Any, *, setter: Setter | None=None):
        """ Sets the value of this property from a JSON value.

        This method first

        Args:
            obj (HasProps) :

            json (JSON-dict) :

            models(seq[Model], optional) :

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
        if isinstance(value, dict):
            old = getattr(obj, self.name)
            if old is not None:
                try:
                    self.property.value_type.validate(old, False)
                    if 'value' in value:
                        value = value['value']
                except ValueError:
                    if isinstance(old, str) and 'field' in value:
                        value = value['field']
        super().set_from_json(obj, value, setter=setter)
from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
class UnitsSpecPropertyDescriptor(DataSpecPropertyDescriptor):
    """ A ``PropertyDescriptor`` for Bokeh ``UnitsSpec`` properties that
    contribute associated ``_units`` properties automatically as a side effect.

    """

    def __init__(self, name, property, units_property) -> None:
        """

        Args:
            name (str) :
                The attribute name that this property is for

            property (Property) :
                A basic property to create a descriptor for

            units_property (Property) :
                An associated property to hold units information

        """
        super().__init__(name, property)
        self.units_prop = units_property

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
        value = self._extract_units(obj, value)
        super().__set__(obj, value, setter=setter)

    def set_from_json(self, obj, json, *, models=None, setter=None):
        """ Sets the value of this property from a JSON value.

        This method first separately extracts and removes any ``units`` field
        in the JSON, and sets the associated units property directly. The
        remaining JSON is then passed to the superclass ``set_from_json`` to
        be handled.

        Args:
            obj: (HasProps) : instance to set the property value on

            json: (JSON-value) : value to set to the attribute to

            models (dict or None, optional) :
                Mapping of model ids to models (default: None)

                This is needed in cases where the attributes to update also
                have values that have references.

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
        json = self._extract_units(obj, json)
        super().set_from_json(obj, json, setter=setter)

    def _extract_units(self, obj, value):
        """ Internal helper for dealing with units associated units properties
        when setting values on ``UnitsSpec`` properties.

        When ``value`` is a dict, this function may mutate the value of the
        associated units property.

        Args:
            obj (HasProps) : instance to update units spec property value for
            value (obj) : new value to set for the property

        Returns:
            copy of ``value``, with 'units' key and value removed when
            applicable

        """
        if isinstance(value, dict):
            if 'units' in value:
                value = copy(value)
            units = value.pop('units', None)
            if units:
                self.units_prop.__set__(obj, units)
        return value
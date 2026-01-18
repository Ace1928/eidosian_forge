from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class Themeables(dict[str, themeable]):
    """
    Collection of themeables

    The key is the name of the class.
    """

    def update(self, other: Themeables, **kwargs):
        """
        Update themeables with those from `other`

        This method takes care of inserting the `themeable`
        into the underlying dictionary. Before doing the
        insertion, any existing themeables that will be
        affected by a new from `other` will either be merged
        or removed. This makes sure that a general themeable
        of type [](`~plotnine.theme.themeables.text`) can be
        added to override an existing specific one of type
        [](`~plotnine.theme.themeables.axis_text_x`).
        """
        for new in other.values():
            new_key = new.__class__.__name__
            for child in new.__class__.mro()[1:-2]:
                child_key = child.__name__
                try:
                    self[child_key].merge(new)
                except KeyError:
                    pass
                except ValueError:
                    del self[child_key]
            try:
                self[new_key].merge(new)
            except (KeyError, ValueError):
                self[new_key] = new

    @property
    def _dict(self):
        """
        Themeables in reverse based on the inheritance hierarchy.

        Themeables should be applied or merged in order from general
        to specific. i.e.
            - apply [](`~plotnine.theme.themeables.axis_line`)
              before [](`~plotnine.theme.themeables.axis_line_x`)
            - merge [](`~plotnine.theme.themeables.axis_line_x`)
              into [](`~plotnine.theme.themeables.axis_line`)
        """
        hierarchy = themeable._hierarchy
        result: dict[str, themeable] = {}
        for lst in hierarchy.values():
            for name in reversed(lst):
                if name in self and name not in result:
                    result[name] = self[name]
        return result

    def setup(self, theme: theme):
        """
        Setup themeables for theming
        """
        for name, th in self.items():
            if isinstance(th.theme_element, element_base):
                th.theme_element.setup(theme, name)

    def items(self):
        """
        List of (name, themeable) in reverse based on the inheritance.
        """
        return self._dict.items()

    def values(self):
        """
        List of themeables in reverse based on the inheritance hierarchy.
        """
        return self._dict.values()

    def getp(self, key: str | tuple[str, str], default: Any=None) -> Any:
        """
        Get the value a specific themeable(s) property

        Themeables store theming attribute values in the
        [](`~plotnine.themes.themeables.Themeable.properties`)
        [](`dict`). The goal of this method is to look a value from
        that dictionary, and fallback along the inheritance heirarchy
        of themeables.

        Parameters
        ----------
        key :
            Themeable and property name to lookup. If a `str`,
            the name is assumed to be "value".

        default :
            Value to return if lookup fails
        Returns
        -------
        out : object
            Value

        Raises
        ------
        KeyError
            If key is in not in any of themeables
        """
        if isinstance(key, str):
            key = (key, 'value')
        name, prop = key
        hlist = themeable._hierarchy[name]
        scalar = key == 'value'
        for th in hlist:
            with suppress(KeyError):
                value = self[th]._properties[prop]
                if not scalar or value is not None:
                    return value
        return default

    def property(self, name: str, key: str='value') -> Any:
        """
        Get the value a specific themeable(s) property

        Themeables store theming attribute values in the
        [](`~plotnine.theme.themeables.Themeable.properties`)
        [](`dict`). The goal of this method is to look a value from
        that dictionary, and fallback along the inheritance heirarchy
        of themeables.

        Parameters
        ----------
        name : str
            Themeable name
        key : str
            Property name to lookup

        Returns
        -------
        out : object
            Value

        Raises
        ------
        KeyError
            If key is in not in any of themeables
        """
        default = object()
        res = self.getp((name, key), default)
        if res is default:
            hlist = themeable._hierarchy[name]
            msg = f"'{key}' is not in the properties of {hlist}"
            raise KeyError(msg)
        return res

    def is_blank(self, name: str) -> bool:
        """
        Return True if the themeable *name* is blank

        If the *name* is not in the list of themeables then
        the lookup falls back to inheritance hierarchy.
        If none of the themeables are in the hierarchy are
        present, `False` is returned.

        Parameters
        ----------
        names : str
            Themeable, in order of most specific to most
            general.
        """
        for th in themeable._hierarchy[name]:
            if (element := self.get(th)):
                return element.is_blank()
        return False
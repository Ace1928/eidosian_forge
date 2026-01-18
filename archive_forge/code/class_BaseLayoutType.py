import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
class BaseLayoutType(BaseLayoutHierarchyType):
    """
    Base class for the layout type. The Layout class itself is a
    code-generated subclass.
    """

    @property
    def _subplotid_validators(self):
        """
        dict of validator classes for each subplot type

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    def _subplot_re_match(self, prop):
        raise NotImplementedError()

    def __init__(self, plotly_name, **kwargs):
        """
        Construct a new BaseLayoutType object

        Parameters
        ----------
        plotly_name : str
            Name of the object (should always be 'layout')
        kwargs : dict[str, any]
            Properties that were not recognized by the Layout subclass.
            These are subplot identifiers (xaxis2, geo4, etc.) or they are
            invalid properties.
        """
        assert plotly_name == 'layout'
        super(BaseLayoutHierarchyType, self).__init__(plotly_name)
        self._subplotid_props = set()
        self._process_kwargs(**kwargs)

    def _process_kwargs(self, **kwargs):
        """
        Process any extra kwargs that are not predefined as constructor params
        """
        unknown_kwargs = {k: v for k, v in kwargs.items() if not self._subplot_re_match(k)}
        super(BaseLayoutHierarchyType, self)._process_kwargs(**unknown_kwargs)
        subplot_kwargs = {k: v for k, v in kwargs.items() if self._subplot_re_match(k)}
        for prop, value in subplot_kwargs.items():
            self._set_subplotid_prop(prop, value)

    def _set_subplotid_prop(self, prop, value):
        """
        Set a subplot property on the layout

        Parameters
        ----------
        prop : str
            A valid subplot property
        value
            Subplot value
        """
        match = self._subplot_re_match(prop)
        subplot_prop = match.group(1)
        suffix_digit = int(match.group(2))
        if suffix_digit == 0:
            raise TypeError('Subplot properties may only be suffixed by an integer >= 1\nReceived {k}'.format(k=prop))
        if suffix_digit == 1:
            prop = subplot_prop
        if prop not in self._valid_props:
            self._valid_props.add(prop)
        self._set_compound_prop(prop, value)
        self._subplotid_props.add(prop)

    def _strip_subplot_suffix_of_1(self, prop):
        """
        Strip the suffix for subplot property names that have a suffix of 1.
        All other properties are returned unchanged

        e.g. 'xaxis1' -> 'xaxis'

        Parameters
        ----------
        prop : str|tuple

        Returns
        -------
        str|tuple
        """
        prop_tuple = BaseFigure._str_to_dict_path(prop)
        if len(prop_tuple) != 1 or not isinstance(prop_tuple[0], str):
            return prop
        else:
            prop = prop_tuple[0]
        match = self._subplot_re_match(prop)
        if match:
            subplot_prop = match.group(1)
            suffix_digit = int(match.group(2))
            if subplot_prop and suffix_digit == 1:
                prop = subplot_prop
        return prop

    def _get_prop_validator(self, prop):
        """
        Custom _get_prop_validator that handles subplot properties
        """
        prop = self._strip_subplot_suffix_of_1(prop)
        return super(BaseLayoutHierarchyType, self)._get_prop_validator(prop)

    def __getattr__(self, prop):
        """
        Custom __getattr__ that handles dynamic subplot properties
        """
        prop = self._strip_subplot_suffix_of_1(prop)
        if prop != '_subplotid_props' and prop in self._subplotid_props:
            validator = self._get_validator(prop)
            return validator.present(self._compound_props[prop])
        else:
            return super(BaseLayoutHierarchyType, self).__getattribute__(prop)

    def __getitem__(self, prop):
        """
        Custom __getitem__ that handles dynamic subplot properties
        """
        prop = self._strip_subplot_suffix_of_1(prop)
        return super(BaseLayoutHierarchyType, self).__getitem__(prop)

    def __contains__(self, prop):
        """
        Custom __contains__ that handles dynamic subplot properties
        """
        prop = self._strip_subplot_suffix_of_1(prop)
        return super(BaseLayoutHierarchyType, self).__contains__(prop)

    def __setitem__(self, prop, value):
        """
        Custom __setitem__ that handles dynamic subplot properties
        """
        prop_tuple = BaseFigure._str_to_dict_path(prop)
        if len(prop_tuple) != 1 or not isinstance(prop_tuple[0], str):
            super(BaseLayoutHierarchyType, self).__setitem__(prop, value)
            return
        else:
            prop = prop_tuple[0]
        match = self._subplot_re_match(prop)
        if match is None:
            super(BaseLayoutHierarchyType, self).__setitem__(prop, value)
        else:
            self._set_subplotid_prop(prop, value)

    def __setattr__(self, prop, value):
        """
        Custom __setattr__ that handles dynamic subplot properties
        """
        match = self._subplot_re_match(prop)
        if match is None:
            super(BaseLayoutHierarchyType, self).__setattr__(prop, value)
        else:
            self._set_subplotid_prop(prop, value)

    def __dir__(self):
        """
        Custom __dir__ that handles dynamic subplot properties
        """
        return list(super(BaseLayoutHierarchyType, self).__dir__()) + sorted(self._subplotid_props)
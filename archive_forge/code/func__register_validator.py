from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
def _register_validator(self, handler: t.Callable[..., None], names: tuple[str | Sentinel, ...]) -> None:
    """Setup a handler to be called when a trait should be cross validated.

        This is used to setup dynamic notifications for cross-validation.

        If a validator is already registered for any of the provided names, a
        TraitError is raised and no new validator is registered.

        Parameters
        ----------
        handler : callable
            A callable that is called when the given trait is cross-validated.
            Its signature is handler(proposal), where proposal is a Bunch (dictionary with attribute access)
            with the following attributes/keys:
                * ``owner`` : the HasTraits instance
                * ``value`` : the proposed value for the modified trait attribute
                * ``trait`` : the TraitType instance associated with the attribute
        names : List of strings
            The names of the traits that should be cross-validated
        """
    for name in names:
        magic_name = '_%s_validate' % name
        if hasattr(self, magic_name):
            class_value = getattr(self.__class__, magic_name)
            if not isinstance(class_value, ValidateHandler):
                deprecated_method(class_value, self.__class__, magic_name, 'use @validate decorator instead.')
    for name in names:
        self._trait_validators[name] = handler
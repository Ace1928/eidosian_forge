from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
def _classes_with_config_traits(self, classes: ClassesType | None=None) -> t.Generator[type[Configurable], None, None]:
    """
        Yields only classes with configurable traits, and their subclasses.

        :param classes:
            The list of classes to iterate; if not set, uses :attr:`classes`.

        Thus, produced sample config-file will contain all classes
        on which a trait-value may be overridden:

        - either on the class owning the trait,
        - or on its subclasses, even if those subclasses do not define
          any traits themselves.
        """
    if classes is None:
        classes = self.classes
    cls_to_config = OrderedDict(((cls, bool(cls.class_own_traits(config=True))) for cls in self._classes_inc_parents(classes)))

    def is_any_parent_included(cls: t.Any) -> bool:
        return any((b in cls_to_config and cls_to_config[b] for b in cls.__bases__))
    while True:
        to_incl_orig = cls_to_config.copy()
        cls_to_config = OrderedDict(((cls, inc_yes or is_any_parent_included(cls)) for cls, inc_yes in cls_to_config.items()))
        if cls_to_config == to_incl_orig:
            break
    for cl, inc_yes in cls_to_config.items():
        if inc_yes:
            yield cl
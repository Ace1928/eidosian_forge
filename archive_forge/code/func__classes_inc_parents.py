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
def _classes_inc_parents(self, classes: ClassesType | None=None) -> t.Generator[type[Configurable], None, None]:
    """Iterate through configurable classes, including configurable parents

        :param classes:
            The list of classes to iterate; if not set, uses :attr:`classes`.

        Children should always be after parents, and each class should only be
        yielded once.
        """
    if classes is None:
        classes = self.classes
    seen = set()
    for c in classes:
        for parent in reversed(c.mro()):
            if issubclass(parent, Configurable) and parent not in seen:
                seen.add(parent)
                yield parent
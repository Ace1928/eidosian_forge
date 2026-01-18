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
class ValidateHandler(EventHandler):

    def __init__(self, names: tuple[Sentinel | str, ...]) -> None:
        self.trait_names = names

    def instance_init(self, inst: HasTraits) -> None:
        inst._register_validator(self, self.trait_names)
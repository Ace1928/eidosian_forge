from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
@observe('config')
@observe_compat
def _config_changed(self, change: Bunch) -> None:
    """Update all the class traits having ``config=True`` in metadata.

        For any class trait with a ``config`` metadata attribute that is
        ``True``, we update the trait with the value of the corresponding
        config entry.
        """
    traits = self.traits(config=True)
    section_names = self.section_names()
    self._load_config(change.new, traits=traits, section_names=section_names)
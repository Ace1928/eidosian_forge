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
def _find_my_config(self, cfg: Config) -> t.Any:
    """extract my config from a global Config object

        will construct a Config object of only the config values that apply to me
        based on my mro(), as well as those of my parent(s) if they exist.

        If I am Bar and my parent is Foo, and their parent is Tim,
        this will return merge following config sections, in this order::

            [Bar, Foo.Bar, Tim.Foo.Bar]

        With the last item being the highest priority.
        """
    cfgs = [cfg]
    if self.parent:
        cfgs.append(self.parent._find_my_config(cfg))
    my_config = Config()
    for c in cfgs:
        for sname in self.section_names():
            if c._has_section(sname):
                my_config.merge(c[sname])
    return my_config
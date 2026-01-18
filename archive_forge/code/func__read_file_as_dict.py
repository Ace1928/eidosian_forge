from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
def _read_file_as_dict(self) -> None:
    """Load the config file into self.config, with recursive loading."""

    def get_config() -> Config:
        """Unnecessary now, but a deprecation warning is more trouble than it's worth."""
        return self.config
    namespace = dict(c=self.config, load_subconfig=self.load_subconfig, get_config=get_config, __file__=self.full_filename)
    conf_filename = self.full_filename
    with open(conf_filename, 'rb') as f:
        exec(compile(f.read(), conf_filename, 'exec'), namespace, namespace)
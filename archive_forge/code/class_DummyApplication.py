import argparse
import inspect
import locale
import os
import pkgutil
import pydoc
import re
import sys
from gettext import NullTranslations
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type
from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autosummary import (ImportExceptionGroup, get_documenter, import_by_name,
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst, split_full_qualified_name
from sphinx.util.inspect import getall, safe_getattr
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxTemplateLoader
class DummyApplication:
    """Dummy Application class for sphinx-autogen command."""

    def __init__(self, translator: NullTranslations) -> None:
        self.config = Config()
        self.registry = SphinxComponentRegistry()
        self.messagelog: List[str] = []
        self.srcdir = '/'
        self.translator = translator
        self.verbosity = 0
        self._warncount = 0
        self.warningiserror = False
        self.config.add('autosummary_context', {}, True, None)
        self.config.add('autosummary_filename_map', {}, True, None)
        self.config.add('autosummary_ignore_module_all', True, 'env', bool)
        self.config.init_values()

    def emit_firstresult(self, *args: Any) -> None:
        pass
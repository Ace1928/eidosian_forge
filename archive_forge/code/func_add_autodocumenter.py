import os
import pickle
import sys
import warnings
from collections import deque
from io import StringIO
from os import path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive, roles
from docutils.transforms import Transform
from pygments.lexer import Lexer
import sphinx
from sphinx import locale, package_dir
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain, Index
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.errors import ApplicationError, ConfigError, VersionRequirementError
from sphinx.events import EventManager
from sphinx.extension import Extension
from sphinx.highlighting import lexer_classes
from sphinx.locale import __
from sphinx.project import Project
from sphinx.registry import SphinxComponentRegistry
from sphinx.roles import XRefRole
from sphinx.theming import Theme
from sphinx.util import docutils, logging, progress_message
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogRepository
from sphinx.util.logging import prefixed_warnings
from sphinx.util.osutil import abspath, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.typing import RoleFunction, TitleGetter
def add_autodocumenter(self, cls: Any, override: bool=False) -> None:
    """Register a new documenter class for the autodoc extension.

        Add *cls* as a new documenter class for the :mod:`sphinx.ext.autodoc`
        extension.  It must be a subclass of
        :class:`sphinx.ext.autodoc.Documenter`.  This allows auto-documenting
        new types of objects.  See the source of the autodoc module for
        examples on how to subclass :class:`Documenter`.

        If *override* is True, the given *cls* is forcedly installed even if
        a documenter having the same name is already installed.

        See :ref:`autodoc_ext_tutorial`.

        .. versionadded:: 0.6
        .. versionchanged:: 2.2
           Add *override* keyword.
        """
    logger.debug('[app] adding autodocumenter: %r', cls)
    from sphinx.ext.autodoc.directive import AutodocDirective
    self.registry.add_documenter(cls.objtype, cls)
    self.add_directive('auto' + cls.objtype, AutodocDirective, override=override)
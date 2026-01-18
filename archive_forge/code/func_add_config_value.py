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
def add_config_value(self, name: str, default: Any, rebuild: Union[bool, str], types: Any=()) -> None:
    """Register a configuration value.

        This is necessary for Sphinx to recognize new values and set default
        values accordingly.


        :param name: The name of the configuration value.  It is recommended to be prefixed
                     with the extension name (ex. ``html_logo``, ``epub_title``)
        :param default: The default value of the configuration.
        :param rebuild: The condition of rebuild.  It must be one of those values:

                        * ``'env'`` if a change in the setting only takes effect when a
                          document is parsed -- this means that the whole environment must be
                          rebuilt.
                        * ``'html'`` if a change in the setting needs a full rebuild of HTML
                          documents.
                        * ``''`` if a change in the setting will not need any special rebuild.
        :param types: The type of configuration value.  A list of types can be specified.  For
                      example, ``[str]`` is used to describe a configuration that takes string
                      value.

        .. versionchanged:: 0.4
           If the *default* value is a callable, it will be called with the
           config object as its argument in order to get the default value.
           This can be used to implement config values whose default depends on
           other values.

        .. versionchanged:: 0.6
           Changed *rebuild* from a simple boolean (equivalent to ``''`` or
           ``'env'``) to a string.  However, booleans are still accepted and
           converted internally.
        """
    logger.debug('[app] adding config value: %r', (name, default, rebuild, types))
    if rebuild in (False, True):
        rebuild = 'env' if rebuild else ''
    self.config.add(name, default, rebuild, types)
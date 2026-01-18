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
def add_enumerable_node(self, node: Type[Element], figtype: str, title_getter: Optional[TitleGetter]=None, override: bool=False, **kwargs: Tuple[Callable, Callable]) -> None:
    """Register a Docutils node class as a numfig target.

        Sphinx numbers the node automatically. And then the users can refer it
        using :rst:role:`numref`.

        :param node: A node class
        :param figtype: The type of enumerable nodes.  Each figtype has individual numbering
                        sequences.  As system figtypes, ``figure``, ``table`` and
                        ``code-block`` are defined.  It is possible to add custom nodes to
                        these default figtypes.  It is also possible to define new custom
                        figtype if a new figtype is given.
        :param title_getter: A getter function to obtain the title of node.  It takes an
                             instance of the enumerable node, and it must return its title as
                             string.  The title is used to the default title of references for
                             :rst:role:`ref`.  By default, Sphinx searches
                             ``docutils.nodes.caption`` or ``docutils.nodes.title`` from the
                             node as a title.
        :param kwargs: Visitor functions for each builder (same as :meth:`add_node`)
        :param override: If true, install the node forcedly even if another node is already
                         installed as the same name

        .. versionadded:: 1.4
        """
    self.registry.add_enumerable_node(node, figtype, title_getter, override=override)
    self.add_node(node, override=override, **kwargs)
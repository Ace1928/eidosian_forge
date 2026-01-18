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
def add_crossref_type(self, directivename: str, rolename: str, indextemplate: str='', ref_nodeclass: Optional[Type[TextElement]]=None, objname: str='', override: bool=False) -> None:
    """Register a new crossref object type.

        This method is very similar to :meth:`add_object_type` except that the
        directive it generates must be empty, and will produce no output.

        That means that you can add semantic targets to your sources, and refer
        to them using custom roles instead of generic ones (like
        :rst:role:`ref`).  Example call::

           app.add_crossref_type('topic', 'topic', 'single: %s',
                                 docutils.nodes.emphasis)

        Example usage::

           .. topic:: application API

           The application API
           -------------------

           Some random text here.

           See also :topic:`this section <application API>`.

        (Of course, the element following the ``topic`` directive needn't be a
        section.)


        :param override: If false, do not install it if another cross-reference type
                         is already installed as the same name
                         If true, unconditionally install the cross-reference type.

        .. versionchanged:: 1.8
           Add *override* keyword.
        """
    self.registry.add_crossref_type(directivename, rolename, indextemplate, ref_nodeclass, objname, override=override)
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Type
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst.states import RSTState
from docutils.statemachine import StringList
from docutils.utils import Reporter, assemble_option_dict
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Documenter, Options
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import nested_parse_with_titles
class DocumenterBridge:
    """A parameters container for Documenters."""

    def __init__(self, env: BuildEnvironment, reporter: Optional[Reporter], options: Options, lineno: int, state: Any) -> None:
        self.env = env
        self._reporter = reporter
        self.genopt = options
        self.lineno = lineno
        self.record_dependencies: Set[str] = set()
        self.result = StringList()
        self.state = state

    def warn(self, msg: str) -> None:
        warnings.warn('DocumenterBridge.warn is deprecated.  Please use sphinx.util.logging module instead.', RemovedInSphinx60Warning, stacklevel=2)
        logger.warning(msg, location=(self.env.docname, self.lineno))

    @property
    def filename_set(self) -> Set:
        warnings.warn('DocumenterBridge.filename_set is deprecated.', RemovedInSphinx60Warning, stacklevel=2)
        return self.record_dependencies
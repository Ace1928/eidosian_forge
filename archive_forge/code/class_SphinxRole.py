import os
import re
import warnings
from contextlib import contextmanager
from copy import copy
from os import path
from types import ModuleType
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set,
import docutils
from docutils import nodes
from docutils.io import FileOutput
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import State, StateMachine, StringList
from docutils.utils import Reporter, unescape
from docutils.writers._html_base import HTMLTranslator
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.typing import RoleFunction
class SphinxRole:
    """A base class for Sphinx roles.

    This class provides helper methods for Sphinx roles.

    .. note:: The subclasses of this class might not work with docutils.
              This class is strongly coupled with Sphinx.
    """
    name: str
    rawtext: str
    text: str
    lineno: int
    inliner: Inliner
    options: Dict
    content: List[str]

    def __call__(self, name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options: Dict={}, content: List[str]=[]) -> Tuple[List[Node], List[system_message]]:
        self.rawtext = rawtext
        self.text = unescape(text)
        self.lineno = lineno
        self.inliner = inliner
        self.options = options
        self.content = content
        if name:
            self.name = name.lower()
        else:
            self.name = self.env.temp_data.get('default_role', '')
            if not self.name:
                self.name = self.env.config.default_role
            if not self.name:
                raise SphinxError('cannot determine default role!')
        return self.run()

    def run(self) -> Tuple[List[Node], List[system_message]]:
        raise NotImplementedError

    @property
    def env(self) -> 'BuildEnvironment':
        """Reference to the :class:`.BuildEnvironment` object."""
        return self.inliner.document.settings.env

    @property
    def config(self) -> 'Config':
        """Reference to the :class:`.Config` object."""
        return self.env.config

    def get_source_info(self, lineno: int=None) -> Tuple[str, int]:
        if lineno is None:
            lineno = self.lineno
        return self.inliner.reporter.get_source_and_line(lineno)

    def set_source_info(self, node: Node, lineno: int=None) -> None:
        node.source, node.line = self.get_source_info(lineno)

    def get_location(self) -> str:
        """Get current location info for logging."""
        return ':'.join((str(s) for s in self.get_source_info()))
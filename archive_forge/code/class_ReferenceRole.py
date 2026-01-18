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
class ReferenceRole(SphinxRole):
    """A base class for reference roles.

    The reference roles can accept ``link title <target>`` style as a text for
    the role.  The parsed result; link title and target will be stored to
    ``self.title`` and ``self.target``.
    """
    has_explicit_title: bool
    disabled: bool
    title: str
    target: str
    explicit_title_re = re.compile('^(.+?)\\s*(?<!\\x00)<(.*?)>$', re.DOTALL)

    def __call__(self, name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options: Dict={}, content: List[str]=[]) -> Tuple[List[Node], List[system_message]]:
        self.disabled = text.startswith('!')
        matched = self.explicit_title_re.match(text)
        if matched:
            self.has_explicit_title = True
            self.title = unescape(matched.group(1))
            self.target = unescape(matched.group(2))
        else:
            self.has_explicit_title = False
            self.title = unescape(text)
            self.target = unescape(text)
        return super().__call__(name, rawtext, text, lineno, inliner, options, content)
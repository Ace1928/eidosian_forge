from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.typing import TextlikeNode
Transform a single field list *node*.
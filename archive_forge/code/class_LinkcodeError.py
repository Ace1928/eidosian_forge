from typing import Any, Dict, Set
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import _
class LinkcodeError(SphinxError):
    category = 'linkcode error'
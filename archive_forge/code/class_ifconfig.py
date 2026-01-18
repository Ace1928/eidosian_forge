from typing import Any, Dict, List
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
class ifconfig(nodes.Element):
    pass
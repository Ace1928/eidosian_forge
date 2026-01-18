import sys
from docutils import nodes, utils
from docutils.parsers.rst import Directive
from docutils.parsers.rst import states
from docutils.transforms import components
class meta(nodes.Special, nodes.PreBibliographic, nodes.Element):
    """HTML-specific "meta" element."""
    pass
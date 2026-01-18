from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class tabular_col_spec(nodes.Element):
    """Node for specifying tabular columns, used for LaTeX output."""
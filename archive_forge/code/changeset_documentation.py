from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
Domain for changesets.
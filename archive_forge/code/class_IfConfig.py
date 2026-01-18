from typing import Any, Dict, List
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
class IfConfig(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {}

    def run(self) -> List[Node]:
        node = ifconfig()
        node.document = self.state.document
        self.set_source_info(node)
        node['expr'] = self.arguments[0]
        nested_parse_with_titles(self.state, self.content, node)
        return [node]
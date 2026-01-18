from typing import Any, Dict, List, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform
class RefOnlyBulletListTransform(SphinxTransform):
    """Change refonly bullet lists to use compact_paragraphs.

    Specifically implemented for 'Indices and Tables' section, which looks
    odd when html_compact_lists is false.
    """
    default_priority = 100

    def apply(self, **kwargs: Any) -> None:
        if self.config.html_compact_lists:
            return

        def check_refonly_list(node: Node) -> bool:
            """Check for list with only references in it."""
            visitor = RefOnlyListChecker(self.document)
            try:
                node.walk(visitor)
            except nodes.NodeFound:
                return False
            else:
                return True
        for node in self.document.findall(nodes.bullet_list):
            if check_refonly_list(node):
                for item in node.findall(nodes.list_item):
                    para = cast(nodes.paragraph, item[0])
                    ref = cast(nodes.reference, para[0])
                    compact_para = addnodes.compact_paragraph()
                    compact_para += ref
                    item.replace(para, compact_para)
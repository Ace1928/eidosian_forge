import re
import sys
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class ContentsFilter(nodes.TreeCopyVisitor):

    def get_entry_text(self):
        return self.get_tree_copy().children

    def visit_citation_reference(self, node):
        raise nodes.SkipNode

    def visit_footnote_reference(self, node):
        raise nodes.SkipNode

    def visit_image(self, node):
        if node.hasattr('alt'):
            self.parent.append(nodes.Text(node['alt']))
        raise nodes.SkipNode

    def ignore_node_but_process_children(self, node):
        raise nodes.SkipDeparture
    visit_problematic = ignore_node_but_process_children
    visit_reference = ignore_node_but_process_children
    visit_target = ignore_node_but_process_children
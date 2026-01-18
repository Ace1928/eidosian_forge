import os.path
import docutils
from docutils import frontend, nodes, writers, io
from docutils.transforms import writer_aux
from docutils.writers import _html_base
class SimpleListChecker(writers._html_base.SimpleListChecker):
    """
    Raise `nodes.NodeFound` if non-simple list item is encountered.

    Here "simple" means a list item containing nothing other than a single
    paragraph, a simple list, or a paragraph followed by a simple list.
    """

    def visit_list_item(self, node):
        children = []
        for child in node.children:
            if not isinstance(child, nodes.Invisible):
                children.append(child)
        if children and isinstance(children[0], nodes.paragraph) and (isinstance(children[-1], nodes.bullet_list) or isinstance(children[-1], nodes.enumerated_list)):
            children.pop()
        if len(children) <= 1:
            return
        else:
            raise nodes.NodeFound

    def visit_paragraph(self, node):
        raise nodes.SkipNode

    def visit_definition_list(self, node):
        raise nodes.NodeFound

    def visit_docinfo(self, node):
        raise nodes.NodeFound
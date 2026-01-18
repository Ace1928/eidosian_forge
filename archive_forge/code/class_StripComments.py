import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
class StripComments(Transform):
    """
    Remove comment elements from the document tree (only if the
    ``strip_comments`` setting is enabled).
    """
    default_priority = 740

    def apply(self):
        if self.document.settings.strip_comments:
            for node in self.document.traverse(nodes.comment):
                node.parent.remove(node)
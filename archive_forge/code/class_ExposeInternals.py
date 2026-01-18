import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
class ExposeInternals(Transform):
    """
    Expose internal attributes if ``expose_internals`` setting is set.
    """
    default_priority = 840

    def not_Text(self, node):
        return not isinstance(node, nodes.Text)

    def apply(self):
        if self.document.settings.expose_internals:
            for node in self.document.traverse(self.not_Text):
                for att in self.document.settings.expose_internals:
                    value = getattr(node, att, None)
                    if value is not None:
                        node['internal:' + att] = value
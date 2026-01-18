import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
class StripClassesAndElements(Transform):
    """
    Remove from the document tree all elements with classes in
    `self.document.settings.strip_elements_with_classes` and all "classes"
    attribute values in `self.document.settings.strip_classes`.
    """
    default_priority = 420

    def apply(self):
        if not (self.document.settings.strip_elements_with_classes or self.document.settings.strip_classes):
            return
        self.strip_elements = dict([(key, None) for key in self.document.settings.strip_elements_with_classes or []])
        self.strip_classes = dict([(key, None) for key in self.document.settings.strip_classes or []])
        for node in self.document.traverse(self.check_classes):
            node.parent.remove(node)

    def check_classes(self, node):
        if isinstance(node, nodes.Element):
            for class_value in node['classes'][:]:
                if class_value in self.strip_classes:
                    node['classes'].remove(class_value)
                if class_value in self.strip_elements:
                    return 1
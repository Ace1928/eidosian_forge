import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
class TestMessages(Transform):
    """
    Append all post-parse system messages to the end of the document.

    Used for testing purposes.
    """
    default_priority = 880

    def apply(self):
        for msg in self.document.transform_messages:
            if not msg.parent:
                self.document += msg
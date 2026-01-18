import os.path
import sys
import docutils
from docutils import languages, Component
from docutils.transforms import universal
class UnfilteredWriter(Writer):
    """
    A writer that passes the document tree on unchanged (e.g. a
    serializer.)

    Documents written by UnfilteredWriters are typically reused at a
    later date using a subclass of `readers.ReReader`.
    """

    def get_transforms(self):
        return Component.get_transforms(self)
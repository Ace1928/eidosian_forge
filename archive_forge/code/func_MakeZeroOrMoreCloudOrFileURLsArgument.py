from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from .tab_complete import CompleterType
@staticmethod
def MakeZeroOrMoreCloudOrFileURLsArgument():
    """Constructs an argument that takes 0 or more Cloud or File URLs."""
    return CommandArgument('file', nargs='*', completer=CompleterType.CLOUD_OR_LOCAL_OBJECT)
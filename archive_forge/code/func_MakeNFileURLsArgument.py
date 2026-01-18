from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from .tab_complete import CompleterType
@staticmethod
def MakeNFileURLsArgument(n):
    """Constructs an argument that takes N File URLs as parameters."""
    return CommandArgument('file', nargs=n, completer=CompleterType.LOCAL_OBJECT)
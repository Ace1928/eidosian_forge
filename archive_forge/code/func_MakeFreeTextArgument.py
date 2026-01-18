from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from .tab_complete import CompleterType
@staticmethod
def MakeFreeTextArgument():
    """Constructs an argument that takes arbitrary text."""
    return CommandArgument('text', completer=CompleterType.NO_OP)
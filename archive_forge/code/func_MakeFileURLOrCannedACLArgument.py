from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from .tab_complete import CompleterType
@staticmethod
def MakeFileURLOrCannedACLArgument():
    """Constructs an argument that takes a File URL or a canned ACL."""
    return CommandArgument('file', nargs=1, completer=CompleterType.LOCAL_OBJECT_OR_CANNED_ACL)
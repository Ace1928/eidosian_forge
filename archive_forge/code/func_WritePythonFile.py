import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def WritePythonFile(file_descriptor, package, version, printer):
    """Write the given extended file descriptor to out."""
    _WriteFile(file_descriptor, package, version, _ProtoRpcPrinter(printer))
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def _Commands(component, depth=3):
    """Yields tuples representing commands.

  To use the command from Python, insert '.' between each element of the tuple.
  To use the command from the command line, insert ' ' between each element of
  the tuple.

  Args:
    component: The component considered to be the root of the yielded commands.
    depth: The maximum depth with which to traverse the member DAG for commands.
  Yields:
    Tuples, each tuple representing one possible command for this CLI.
    Only traverses the member DAG up to a depth of depth.
  """
    if inspect.isroutine(component) or inspect.isclass(component):
        for completion in Completions(component, verbose=False):
            yield (completion,)
    if inspect.isroutine(component):
        return
    if depth < 1:
        return
    for member_name, member in VisibleMembers(component, class_attrs={}, verbose=False):
        member_name = _FormatForCommand(member_name)
        yield (member_name,)
        for command in _Commands(member, depth - 1):
            yield ((member_name,) + command)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def VisibleMembers(component, class_attrs=None, verbose=False):
    """Returns a list of the members of the given component.

  If verbose is True, then members starting with _ (normally ignored) are
  included.

  Args:
    component: The component whose members to list.
    class_attrs: (optional) If component is a class, you may provide this as:
      inspectutils.GetClassAttrsDict(component). If not provided, it will be
      computed. If provided, this determines how class members will be treated
      for visibility. In particular, methods are generally hidden for
      non-instantiated classes, but if you wish them to be shown (e.g. for
      completion scripts) then pass in a different class_attr for them.
    verbose: Whether to include private members.
  Returns:
    A list of tuples (member_name, member) of all members of the component.
  """
    if isinstance(component, dict):
        members = component.items()
    else:
        members = inspect.getmembers(component)
    if class_attrs is None:
        class_attrs = inspectutils.GetClassAttrsDict(component)
    return [(member_name, member) for member_name, member in members if MemberVisible(component, member_name, member, class_attrs=class_attrs, verbose=verbose)]
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def NeedsSeparator(self):
    """Returns whether a separator should be added to the command.

    If the command is a function call, then adding an additional argument to the
    command sometimes would add an extra arg to the function call, and sometimes
    would add an arg acting on the result of the function call.

    This function tells us whether we should add a separator to the command
    before adding additional arguments in order to make sure the arg is applied
    to the result of the function call, and not the function call itself.

    Returns:
      Whether a separator should be added to the command if order to keep the
      component referred to by the command the same when adding additional args.
    """
    element = self.GetLastHealthyElement()
    return element.HasCapacity() and (not element.HasSeparator())
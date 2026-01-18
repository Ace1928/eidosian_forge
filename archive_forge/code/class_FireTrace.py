from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
class FireTrace(object):
    """A FireTrace represents the steps taken during a single Fire execution.

  A FireTrace consists of a sequence of FireTraceElement objects. Each element
  represents an action taken by Fire during a single Fire execution. An action
  may be instantiating a class, calling a routine, or accessing a property.
  """

    def __init__(self, initial_component, name=None, separator='-', verbose=False, show_help=False, show_trace=False):
        initial_trace_element = FireTraceElement(component=initial_component, action=INITIAL_COMPONENT)
        self.name = name
        self.separator = separator
        self.elements = [initial_trace_element]
        self.verbose = verbose
        self.show_help = show_help
        self.show_trace = show_trace

    def GetResult(self):
        """Returns the component from the last element of the trace."""
        return self.GetLastHealthyElement().component

    def GetLastHealthyElement(self):
        """Returns the last element of the trace that is not an error.

    This element will contain the final component indicated by the trace.

    Returns:
      The last element of the trace that is not an error.
    """
        for element in reversed(self.elements):
            if not element.HasError():
                return element
        return None

    def HasError(self):
        """Returns whether the Fire execution encountered a Fire usage error."""
        return self.elements[-1].HasError()

    def AddAccessedProperty(self, component, target, args, filename, lineno):
        element = FireTraceElement(component=component, action=ACCESSED_PROPERTY, target=target, args=args, filename=filename, lineno=lineno)
        self.elements.append(element)

    def AddCalledComponent(self, component, target, args, filename, lineno, capacity, action=CALLED_CALLABLE):
        """Adds an element to the trace indicating that a component was called.

    Also applies to instantiating a class.

    Args:
      component: The result of calling the callable.
      target: The name of the callable.
      args: The args consumed in order to call this callable.
      filename: The file in which the callable is defined, or None if N/A.
      lineno: The line number on which the callable is defined, or None if N/A.
      capacity: (bool) Whether the callable could have accepted additional args.
      action: The value to include as the action in the FireTraceElement.
    """
        element = FireTraceElement(component=component, action=action, target=target, args=args, filename=filename, lineno=lineno, capacity=capacity)
        self.elements.append(element)

    def AddCompletionScript(self, script):
        element = FireTraceElement(component=script, action=COMPLETION_SCRIPT)
        self.elements.append(element)

    def AddInteractiveMode(self):
        element = FireTraceElement(action=INTERACTIVE_MODE)
        self.elements.append(element)

    def AddError(self, error, args):
        element = FireTraceElement(error=error, args=args)
        self.elements.append(element)

    def AddSeparator(self):
        """Marks that the most recent element of the trace used  a separator.

    A separator is an argument you can pass to a Fire CLI to separate args left
    of the separator from args right of the separator.

    Here's an example to demonstrate the separator. Let's say you have a
    function that takes a variable number of args, and you want to call that
    function, and then upper case the result. Here's how to do it:

    # in Python
    def display(arg1, arg2='!'):
      return arg1 + arg2

    # from Bash (the default separator is the hyphen -)
    display hello   # hello!
    display hello upper # helloupper
    display hello - upper # HELLO!

    Note how the separator caused the display function to be called with the
    default value for arg2.
    """
        self.elements[-1].AddSeparator()

    def _Quote(self, arg):
        if arg.startswith('--') and '=' in arg:
            prefix, value = arg.split('=', 1)
            return pipes.quote(prefix) + '=' + pipes.quote(value)
        return pipes.quote(arg)

    def GetCommand(self, include_separators=True):
        """Returns the command representing the trace up to this point.

    Args:
      include_separators: Whether or not to include separators in the command.

    Returns:
      A string representing a Fire CLI command that would produce this trace.
    """
        args = []
        if self.name:
            args.append(self.name)
        for element in self.elements:
            if element.HasError():
                continue
            if element.args:
                args.extend(element.args)
            if element.HasSeparator() and include_separators:
                args.append(self.separator)
        if self.NeedsSeparator() and include_separators:
            args.append(self.separator)
        return ' '.join((self._Quote(arg) for arg in args))

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

    def __str__(self):
        lines = []
        for index, element in enumerate(self.elements):
            line = '{index}. {trace_string}'.format(index=index + 1, trace_string=element)
            lines.append(line)
        return '\n'.join(lines)

    def NeedsSeparatingHyphenHyphen(self, flag='help'):
        """Returns whether a the trace need '--' before '--help'.

    '--' is needed when the component takes keyword arguments, when the value of
    flag matches one of the argument of the component, or the component takes in
    keyword-only arguments(e.g. argument with default value).

    Args:
      flag: the flag available for the trace

    Returns:
      True for needed '--', False otherwise.

    """
        element = self.GetLastHealthyElement()
        component = element.component
        spec = inspectutils.GetFullArgSpec(component)
        return spec.varkw is not None or flag in spec.args or flag in spec.kwonlyargs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
def Fire(component=None, command=None, name=None, serialize=None):
    """This function, Fire, is the main entrypoint for Python Fire.

  Executes a command either from the `command` argument or from sys.argv by
  recursively traversing the target object `component`'s members consuming
  arguments, evaluating functions, and instantiating classes as it goes.

  When building a CLI with Fire, your main method should call this function.

  Args:
    component: The initial target component.
    command: Optional. If supplied, this is the command executed. If not
        supplied, then the command is taken from sys.argv instead. This can be
        a string or a list of strings; a list of strings is preferred.
    name: Optional. The name of the command as entered at the command line.
        Used in interactive mode and for generating the completion script.
    serialize: Optional. If supplied, all objects are serialized to text via
        the provided callable.
  Returns:
    The result of executing the Fire command. Execution begins with the initial
    target component. The component is updated by using the command arguments
    to either access a member of the current component, call the current
    component (if it's a function), or instantiate the current component (if
    it's a class). When all arguments are consumed and there's no function left
    to call or class left to instantiate, the resulting current component is
    the final result.
  Raises:
    ValueError: If the command argument is supplied, but not a string or a
        sequence of arguments.
    FireExit: When Fire encounters a FireError, Fire will raise a FireExit with
        code 2. When used with the help or trace flags, Fire will raise a
        FireExit with code 0 if successful.
  """
    name = name or os.path.basename(sys.argv[0])
    if isinstance(command, six.string_types):
        args = shlex.split(command)
    elif isinstance(command, (list, tuple)):
        args = command
    elif command is None:
        args = sys.argv[1:]
    else:
        raise ValueError('The command argument must be a string or a sequence of arguments.')
    args, flag_args = parser.SeparateFlagArgs(args)
    argparser = parser.CreateParser()
    parsed_flag_args, unused_args = argparser.parse_known_args(flag_args)
    context = {}
    if parsed_flag_args.interactive or component is None:
        caller = inspect.stack()[1]
        caller_frame = caller[0]
        caller_globals = caller_frame.f_globals
        caller_locals = caller_frame.f_locals
        context.update(caller_globals)
        context.update(caller_locals)
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
    if component_trace.HasError():
        _DisplayError(component_trace)
        raise FireExit(2, component_trace)
    if component_trace.show_trace and component_trace.show_help:
        output = ['Fire trace:\n{trace}\n'.format(trace=component_trace)]
        result = component_trace.GetResult()
        help_text = helptext.HelpText(result, trace=component_trace, verbose=component_trace.verbose)
        output.append(help_text)
        Display(output, out=sys.stderr)
        raise FireExit(0, component_trace)
    if component_trace.show_trace:
        output = ['Fire trace:\n{trace}'.format(trace=component_trace)]
        Display(output, out=sys.stderr)
        raise FireExit(0, component_trace)
    if component_trace.show_help:
        result = component_trace.GetResult()
        help_text = helptext.HelpText(result, trace=component_trace, verbose=component_trace.verbose)
        output = [help_text]
        Display(output, out=sys.stderr)
        raise FireExit(0, component_trace)
    _PrintResult(component_trace, verbose=component_trace.verbose, serialize=serialize)
    result = component_trace.GetResult()
    return result
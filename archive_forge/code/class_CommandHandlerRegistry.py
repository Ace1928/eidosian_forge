import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class CommandHandlerRegistry:
    """Registry of command handlers for CLI.

  Handler methods (callables) for user commands can be registered with this
  class, which then is able to dispatch commands to the correct handlers and
  retrieve the RichTextLines output.

  For example, suppose you have the following handler defined:
    def echo(argv, screen_info=None):
      return RichTextLines(["arguments = %s" % " ".join(argv),
                            "screen_info = " + repr(screen_info)])

  you can register the handler with the command prefix "echo" and alias "e":
    registry = CommandHandlerRegistry()
    registry.register_command_handler("echo", echo,
        "Echo arguments, along with screen info", prefix_aliases=["e"])

  then to invoke this command handler with some arguments and screen_info, do:
    registry.dispatch_command("echo", ["foo", "bar"], screen_info={"cols": 80})

  or with the prefix alias:
    registry.dispatch_command("e", ["foo", "bar"], screen_info={"cols": 80})

  The call will return a RichTextLines object which can be rendered by a CLI.
  """
    HELP_COMMAND = 'help'
    HELP_COMMAND_ALIASES = ['h']
    VERSION_COMMAND = 'version'
    VERSION_COMMAND_ALIASES = ['ver']

    def __init__(self):
        self._handlers = {}
        self._alias_to_prefix = {}
        self._prefix_to_aliases = {}
        self._prefix_to_help = {}
        self._help_intro = None
        self.register_command_handler(self.HELP_COMMAND, self._help_handler, 'Print this help message.', prefix_aliases=self.HELP_COMMAND_ALIASES)
        self.register_command_handler(self.VERSION_COMMAND, self._version_handler, 'Print the versions of TensorFlow and its key dependencies.', prefix_aliases=self.VERSION_COMMAND_ALIASES)

    def register_command_handler(self, prefix, handler, help_info, prefix_aliases=None):
        """Register a callable as a command handler.

    Args:
      prefix: Command prefix, i.e., the first word in a command, e.g.,
        "print" as in "print tensor_1".
      handler: A callable of the following signature:
          foo_handler(argv, screen_info=None),
        where argv is the argument vector (excluding the command prefix) and
          screen_info is a dictionary containing information about the screen,
          such as number of columns, e.g., {"cols": 100}.
        The callable should return:
          1) a RichTextLines object representing the screen output.

        The callable can also raise an exception of the type CommandLineExit,
        which if caught by the command-line interface, will lead to its exit.
        The exception can optionally carry an exit token of arbitrary type.
      help_info: A help string.
      prefix_aliases: Aliases for the command prefix, as a list of str. E.g.,
        shorthands for the command prefix: ["p", "pr"]

    Raises:
      ValueError: If
        1) the prefix is empty, or
        2) handler is not callable, or
        3) a handler is already registered for the prefix, or
        4) elements in prefix_aliases clash with existing aliases.
        5) help_info is not a str.
    """
        if not prefix:
            raise ValueError('Empty command prefix')
        if prefix in self._handlers:
            raise ValueError('A handler is already registered for command prefix "%s"' % prefix)
        if not callable(handler):
            raise ValueError('handler is not callable')
        if not isinstance(help_info, str):
            raise ValueError('help_info is not a str')
        if prefix_aliases:
            for alias in prefix_aliases:
                if self._resolve_prefix(alias):
                    raise ValueError('The prefix alias "%s" clashes with existing prefixes or aliases.' % alias)
                self._alias_to_prefix[alias] = prefix
            self._prefix_to_aliases[prefix] = prefix_aliases
        self._handlers[prefix] = handler
        self._prefix_to_help[prefix] = help_info

    def dispatch_command(self, prefix, argv, screen_info=None):
        """Handles a command by dispatching it to a registered command handler.

    Args:
      prefix: Command prefix, as a str, e.g., "print".
      argv: Command argument vector, excluding the command prefix, represented
        as a list of str, e.g.,
        ["tensor_1"]
      screen_info: A dictionary containing screen info, e.g., {"cols": 100}.

    Returns:
      An instance of RichTextLines or None. If any exception is caught during
      the invocation of the command handler, the RichTextLines will wrap the
      error type and message.

    Raises:
      ValueError: If
        1) prefix is empty, or
        2) no command handler is registered for the command prefix, or
        3) the handler is found for the prefix, but it fails to return a
          RichTextLines or raise any exception.
      CommandLineExit:
        If the command handler raises this type of exception, this method will
        simply pass it along.
    """
        if not prefix:
            raise ValueError('Prefix is empty')
        resolved_prefix = self._resolve_prefix(prefix)
        if not resolved_prefix:
            raise ValueError('No handler is registered for command prefix "%s"' % prefix)
        handler = self._handlers[resolved_prefix]
        try:
            output = handler(argv, screen_info=screen_info)
        except CommandLineExit as e:
            raise e
        except SystemExit as e:
            lines = ['Syntax error for command: %s' % prefix, 'For help, do "help %s"' % prefix]
            output = RichTextLines(lines)
        except BaseException as e:
            lines = ['Error occurred during handling of command: %s %s:' % (resolved_prefix, ' '.join(argv)), '%s: %s' % (type(e), str(e))]
            lines.append('')
            lines.extend(traceback.format_exc().split('\n'))
            output = RichTextLines(lines)
        if not isinstance(output, RichTextLines) and output is not None:
            raise ValueError('Return value from command handler %s is not None or a RichTextLines instance' % str(handler))
        return output

    def is_registered(self, prefix):
        """Test if a command prefix or its alias is has a registered handler.

    Args:
      prefix: A prefix or its alias, as a str.

    Returns:
      True iff a handler is registered for prefix.
    """
        return self._resolve_prefix(prefix) is not None

    def get_help(self, cmd_prefix=None):
        """Compile help information into a RichTextLines object.

    Args:
      cmd_prefix: Optional command prefix. As the prefix itself or one of its
        aliases.

    Returns:
      A RichTextLines object containing the help information. If cmd_prefix
      is None, the return value will be the full command-line help. Otherwise,
      it will be the help information for the specified command.
    """
        if not cmd_prefix:
            help_info = RichTextLines([])
            if self._help_intro:
                help_info.extend(self._help_intro)
            sorted_prefixes = sorted(self._handlers)
            for cmd_prefix in sorted_prefixes:
                lines = self._get_help_for_command_prefix(cmd_prefix)
                lines.append('')
                lines.append('')
                help_info.extend(RichTextLines(lines))
            return help_info
        else:
            return RichTextLines(self._get_help_for_command_prefix(cmd_prefix))

    def set_help_intro(self, help_intro):
        """Set an introductory message to help output.

    Args:
      help_intro: (RichTextLines) Rich text lines appended to the
        beginning of the output of the command "help", as introductory
        information.
    """
        self._help_intro = help_intro

    def _help_handler(self, args, screen_info=None):
        """Command handler for "help".

    "help" is a common command that merits built-in support from this class.

    Args:
      args: Command line arguments to "help" (not including "help" itself).
      screen_info: (dict) Information regarding the screen, e.g., the screen
        width in characters: {"cols": 80}

    Returns:
      (RichTextLines) Screen text output.
    """
        _ = screen_info
        if not args:
            return self.get_help()
        elif len(args) == 1:
            return self.get_help(args[0])
        else:
            return RichTextLines(['ERROR: help takes only 0 or 1 input argument.'])

    def _version_handler(self, args, screen_info=None):
        del args
        del screen_info
        return get_tensorflow_version_lines(include_dependency_versions=True)

    def _resolve_prefix(self, token):
        """Resolve command prefix from the prefix itself or its alias.

    Args:
      token: a str to be resolved.

    Returns:
      If resolvable, the resolved command prefix.
      If not resolvable, None.
    """
        if token in self._handlers:
            return token
        elif token in self._alias_to_prefix:
            return self._alias_to_prefix[token]
        else:
            return None

    def _get_help_for_command_prefix(self, cmd_prefix):
        """Compile the help information for a given command prefix.

    Args:
      cmd_prefix: Command prefix, as the prefix itself or one of its
        aliases.

    Returns:
      A list of str as the help information fo cmd_prefix. If the cmd_prefix
        does not exist, the returned list of str will indicate that.
    """
        lines = []
        resolved_prefix = self._resolve_prefix(cmd_prefix)
        if not resolved_prefix:
            lines.append('Invalid command prefix: "%s"' % cmd_prefix)
            return lines
        lines.append(resolved_prefix)
        if resolved_prefix in self._prefix_to_aliases:
            lines.append(HELP_INDENT + 'Aliases: ' + ', '.join(self._prefix_to_aliases[resolved_prefix]))
        lines.append('')
        help_lines = self._prefix_to_help[resolved_prefix].split('\n')
        for line in help_lines:
            lines.append(HELP_INDENT + line)
        return lines
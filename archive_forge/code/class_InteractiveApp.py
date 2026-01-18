import itertools
import shlex
import sys
import autopage.argparse
import cmd2
class InteractiveApp(cmd2.Cmd):
    """Provides "interactive mode" features.

    Refer to the cmd2_ and cmd_ documentation for details
    about subclassing and configuring this class.

    .. _cmd2: https://cmd2.readthedocs.io/en/latest/
    .. _cmd: http://docs.python.org/library/cmd.html

    :param parent_app: The calling application (expected to be derived
                       from :class:`cliff.main.App`).
    :param command_manager: A :class:`cliff.commandmanager.CommandManager`
                            instance.
    :param stdin: Standard input stream
    :param stdout: Standard output stream
    """
    use_rawinput = True
    doc_header = 'Shell commands (type help <topic>):'
    app_cmd_header = 'Application commands (type help <topic>):'

    def __init__(self, parent_app, command_manager, stdin, stdout, errexit=False):
        self.parent_app = parent_app
        if not hasattr(sys.stdin, 'isatty') or sys.stdin.isatty():
            self.prompt = '(%s) ' % parent_app.NAME
        else:
            self.prompt = ''
        self.command_manager = command_manager
        self.errexit = errexit
        cmd2.Cmd.__init__(self, 'tab', stdin=stdin, stdout=stdout)

    def _split_line(self, line):
        try:
            return shlex.split(line.parsed.raw)
        except AttributeError:
            parts = shlex.split(line)
            if getattr(line, 'command', None):
                parts.insert(0, line.command)
            return parts

    def default(self, line):
        line_parts = self._split_line(line)
        ret = self.parent_app.run_subcommand(line_parts)
        if self.errexit:
            return ret

    def completenames(self, text, line, begidx, endidx):
        """Tab-completion for command prefix without completer delimiter.

        This method returns cmd style and cliff style commands matching
        provided command prefix (text).
        """
        completions = cmd2.Cmd.completenames(self, text, line, begidx, endidx)
        completions += self._complete_prefix(text)
        return completions

    def completedefault(self, text, line, begidx, endidx):
        """Default tab-completion for command prefix with completer delimiter.

        This method filters only cliff style commands matching provided
        command prefix (line) as cmd2 style commands cannot contain spaces.
        This method returns text + missing command part of matching commands.
        This method does not handle options in cmd2/cliff style commands, you
        must define complete_$method to handle them.
        """
        return [x[begidx:] for x in self._complete_prefix(line)]

    def _complete_prefix(self, prefix):
        """Returns cliff style commands with a specific prefix."""
        if not prefix:
            return [n for n, v in self.command_manager]
        return [n for n, v in self.command_manager if n.startswith(prefix)]

    def help_help(self):
        self.default('help help')

    def do_help(self, arg):
        if arg:
            arg_parts = shlex.split(arg)
            method_name = '_'.join(itertools.chain(['do'], itertools.takewhile(lambda x: not x.startswith('-'), arg_parts)))
            if hasattr(self, method_name):
                return cmd2.Cmd.do_help(self, arg)
            try:
                parsed = self.parsed
            except AttributeError:
                try:
                    parsed = self.parser_manager.parsed
                except AttributeError:
                    parsed = lambda x: x
            self.default(parsed('help ' + arg))
        else:
            stdout = self.stdout
            try:
                with autopage.argparse.help_pager(stdout) as paged_out:
                    self.stdout = paged_out
                    cmd2.Cmd.do_help(self, arg)
                    cmd_names = sorted([n for n, v in self.command_manager])
                    self.print_topics(self.app_cmd_header, cmd_names, 15, 80)
            finally:
                self.stdout = stdout
        return
    do_exit = cmd2.Cmd.do_quit

    def get_names(self):
        return [n for n in cmd2.Cmd.get_names(self) if not n.startswith('do__')]

    def precmd(self, statement):
        """Hook method executed just before the command is executed by
        :meth:`~cmd2.Cmd.onecmd` and after adding it to history.

        :param statement: subclass of str which also contains the parsed input
        :return: a potentially modified version of the input Statement object
        """
        line_parts = self._split_line(statement)
        try:
            the_cmd = self.command_manager.find_command(line_parts)
            cmd_factory, cmd_name, sub_argv = the_cmd
        except ValueError:
            pass
        else:
            if hasattr(statement, 'parsed'):
                statement.parsed.command = cmd_name
                statement.parsed.args = ' '.join(sub_argv)
            else:
                statement = cmd2.Statement(' '.join(sub_argv), raw=statement.raw, command=cmd_name, arg_list=sub_argv, multiline_command=statement.multiline_command, terminator=statement.terminator, suffix=statement.suffix, pipe_to=statement.pipe_to, output=statement.output, output_to=statement.output_to)
        return statement

    def cmdloop(self):
        return self._cmdloop()
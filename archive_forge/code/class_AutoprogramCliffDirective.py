import argparse
import fnmatch
import importlib
import inspect
import re
import sys
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from cliff import app
from cliff import commandmanager
class AutoprogramCliffDirective(rst.Directive):
    """Auto-document a subclass of `cliff.command.Command`."""
    has_content = False
    required_arguments = 1
    option_spec = {'command': directives.unchanged, 'arguments': directives.unchanged, 'ignored': directives.unchanged, 'application': directives.unchanged}

    def _get_ignored_opts(self):
        global_ignored = self.env.config.autoprogram_cliff_ignored
        local_ignored = self.options.get('ignored', '')
        local_ignored = [x.strip() for x in local_ignored.split(',') if x.strip()]
        return list(set(global_ignored + local_ignored))

    def _drop_ignored_options(self, parser, ignored_opts):
        for action in list(parser._actions):
            for option_string in action.option_strings:
                if option_string in ignored_opts:
                    del parser._actions[parser._actions.index(action)]
                    break

    def _load_app(self):
        mod_str, _sep, class_str = self.arguments[0].rpartition('.')
        if not mod_str:
            return
        try:
            importlib.import_module(mod_str)
        except ImportError:
            return
        try:
            cliff_app_class = getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            return
        if not inspect.isclass(cliff_app_class):
            return
        if not issubclass(cliff_app_class, app.App):
            return
        app_arguments = self.options.get('arguments', '').split()
        return cliff_app_class(*app_arguments)

    def _load_command(self, manager, command_name):
        """Load a command using an instance of a `CommandManager`."""
        try:
            return manager.find_command(command_name.split())[0]
        except ValueError:
            raise self.error('"{}" is not a valid command in the "{}" namespace'.format(command_name, manager.namespace))

    def _load_commands(self):
        command_pattern = self.options.get('command')
        manager = commandmanager.CommandManager(self.arguments[0])
        if command_pattern:
            commands = [x for x in manager.commands if fnmatch.fnmatch(x, command_pattern)]
        else:
            commands = manager.commands.keys()
        if not commands:
            msg = 'No commands found in the "{}" namespace'
            if command_pattern:
                msg += ' using the "{}" command name/pattern'
            msg += '. Are you sure this is correct and the application being documented is installed?'
            raise self.warning(msg.format(self.arguments[0], command_pattern))
        return dict(((name, self._load_command(manager, name)) for name in commands))

    def _generate_app_node(self, app, application_name):
        ignored_opts = self._get_ignored_opts()
        parser = app.parser
        self._drop_ignored_options(parser, ignored_opts)
        parser.prog = application_name
        source_name = '<{}>'.format(app.__class__.__name__)
        result = statemachine.ViewList()
        for line in _format_parser(parser):
            result.append(line, source_name)
        section = nodes.section()
        self.state.nested_parse(result, 0, section)
        return section.children

    def _generate_nodes_per_command(self, title, command_name, command_class, ignored_opts):
        """Generate the relevant Sphinx nodes.

        This doesn't bother using raw docutils nodes as they simply don't offer
        the power of directives, like Sphinx's 'option' directive. Instead, we
        generate reStructuredText and parse this in a nested context (to obtain
        correct header levels). Refer to [1] for more information.

        [1] http://www.sphinx-doc.org/en/stable/extdev/markupapi.html

        :param title: Title of command
        :param command_name: Name of command, as used on the command line
        :param command_class: Subclass of :py:class:`cliff.command.Command`
        :param prefix: Prefix to apply before command, if any
        :param ignored_opts: A list of options to exclude from output, if any
        :returns: A list of nested docutil nodes
        """
        command = command_class(None, None)
        if not getattr(command, 'app_dist_name', None):
            command.app_dist_name = self.env.config.autoprogram_cliff_app_dist_name
        parser = command.get_parser(command_name)
        ignored_opts = ignored_opts or []
        self._drop_ignored_options(parser, ignored_opts)
        section = nodes.section('', nodes.title(text=title), ids=[nodes.make_id(title)], names=[nodes.fully_normalize_name(title)])
        source_name = '<{}>'.format(command.__class__.__name__)
        result = statemachine.ViewList()
        for line in _format_parser(parser):
            result.append(line, source_name)
        self.state.nested_parse(result, 0, section)
        return [section]

    def _generate_command_nodes(self, commands, application_name):
        ignored_opts = self._get_ignored_opts()
        output = []
        for command_name in sorted(commands):
            command_class = commands[command_name]
            title = command_name
            if application_name:
                command_name = ' '.join([application_name, command_name])
            output.extend(self._generate_nodes_per_command(title, command_name, command_class, ignored_opts))
        return output

    def run(self):
        self.env = self.state.document.settings.env
        application_name = self.options.get('application') or self.env.config.autoprogram_cliff_application
        app = self._load_app()
        if app:
            return self._generate_app_node(app, application_name)
        commands = self._load_commands()
        return self._generate_command_nodes(commands, application_name)
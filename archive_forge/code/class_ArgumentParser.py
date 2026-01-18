import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class ArgumentParser(_AttributeHolder, _ActionsContainer):
    """Object for parsing command line strings into Python objects.

    Keyword Arguments:
        - prog -- The name of the program (default: sys.argv[0])
        - usage -- A usage message (default: auto-generated from arguments)
        - description -- A description of what the program does
        - epilog -- Text following the argument descriptions
        - parents -- Parsers whose arguments should be copied into this one
        - formatter_class -- HelpFormatter class for printing help messages
        - prefix_chars -- Characters that prefix optional arguments
        - fromfile_prefix_chars -- Characters that prefix files containing
            additional arguments
        - argument_default -- The default value for all arguments
        - conflict_handler -- String indicating how to handle conflicts
        - add_help -- Add a -h/-help option
    """

    def __init__(self, prog=None, usage=None, description=None, epilog=None, version=None, parents=[], formatter_class=HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True):
        if version is not None:
            import warnings
            warnings.warn('The "version" argument to ArgumentParser is deprecated. Please use "add_argument(..., action=\'version\', version="N", ...)" instead', DeprecationWarning)
        superinit = super(ArgumentParser, self).__init__
        superinit(description=description, prefix_chars=prefix_chars, argument_default=argument_default, conflict_handler=conflict_handler)
        if prog is None:
            prog = _os.path.basename(_sys.argv[0])
        self.prog = prog
        self.usage = usage
        self.epilog = epilog
        self.version = version
        self.formatter_class = formatter_class
        self.fromfile_prefix_chars = fromfile_prefix_chars
        self.add_help = add_help
        add_group = self.add_argument_group
        self._positionals = add_group(_('positional arguments'))
        self._optionals = add_group(_('optional arguments'))
        self._subparsers = None

        def identity(string):
            return string
        self.register('type', None, identity)
        if '-' in prefix_chars:
            default_prefix = '-'
        else:
            default_prefix = prefix_chars[0]
        if self.add_help:
            self.add_argument(default_prefix + 'h', default_prefix * 2 + 'help', action='help', default=SUPPRESS, help=_('show this help message and exit'))
        if self.version:
            self.add_argument(default_prefix + 'v', default_prefix * 2 + 'version', action='version', default=SUPPRESS, version=self.version, help=_("show program's version number and exit"))
        for parent in parents:
            self._add_container_actions(parent)
            try:
                defaults = parent._defaults
            except AttributeError:
                pass
            else:
                self._defaults.update(defaults)

    def _get_kwargs(self):
        names = ['prog', 'usage', 'description', 'version', 'formatter_class', 'conflict_handler', 'add_help']
        return [(name, getattr(self, name)) for name in names]

    def add_subparsers(self, **kwargs):
        if self._subparsers is not None:
            self.error(_('cannot have multiple subparser arguments'))
        kwargs.setdefault('parser_class', type(self))
        if 'title' in kwargs or 'description' in kwargs:
            title = _(kwargs.pop('title', 'subcommands'))
            description = _(kwargs.pop('description', None))
            self._subparsers = self.add_argument_group(title, description)
        else:
            self._subparsers = self._positionals
        if kwargs.get('prog') is None:
            formatter = self._get_formatter()
            positionals = self._get_positional_actions()
            groups = self._mutually_exclusive_groups
            formatter.add_usage(self.usage, positionals, groups, '')
            kwargs['prog'] = formatter.format_help().strip()
        parsers_class = self._pop_action_class(kwargs, 'parsers')
        action = parsers_class(option_strings=[], **kwargs)
        self._subparsers._add_action(action)
        return action

    def _add_action(self, action):
        if action.option_strings:
            self._optionals._add_action(action)
        else:
            self._positionals._add_action(action)
        return action

    def _get_optional_actions(self):
        return [action for action in self._actions if action.option_strings]

    def _get_positional_actions(self):
        return [action for action in self._actions if not action.option_strings]

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        return args

    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            args = _sys.argv[1:]
        if namespace is None:
            namespace = Namespace()
        for action in self._actions:
            if action.dest is not SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not SUPPRESS:
                        setattr(namespace, action.dest, action.default)
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])
        try:
            namespace, args = self._parse_known_args(args, namespace)
            if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
                delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
            return (namespace, args)
        except ArgumentError:
            err = _sys.exc_info()[1]
            self.error(str(err))

    def _parse_known_args(self, arg_strings, namespace):
        if self.fromfile_prefix_chars is not None:
            arg_strings = self._read_args_from_files(arg_strings)
        action_conflicts = {}
        for mutex_group in self._mutually_exclusive_groups:
            group_actions = mutex_group._group_actions
            for i, mutex_action in enumerate(mutex_group._group_actions):
                conflicts = action_conflicts.setdefault(mutex_action, [])
                conflicts.extend(group_actions[:i])
                conflicts.extend(group_actions[i + 1:])
        option_string_indices = {}
        arg_string_pattern_parts = []
        arg_strings_iter = iter(arg_strings)
        for i, arg_string in enumerate(arg_strings_iter):
            if arg_string == '--':
                arg_string_pattern_parts.append('-')
                for arg_string in arg_strings_iter:
                    arg_string_pattern_parts.append('A')
            else:
                option_tuple = self._parse_optional(arg_string)
                if option_tuple is None:
                    pattern = 'A'
                else:
                    option_string_indices[i] = option_tuple
                    pattern = 'O'
                arg_string_pattern_parts.append(pattern)
        arg_strings_pattern = ''.join(arg_string_pattern_parts)
        seen_actions = set()
        seen_non_default_actions = set()

        def take_action(action, argument_strings, option_string=None):
            seen_actions.add(action)
            argument_values = self._get_values(action, argument_strings)
            if argument_values is not action.default:
                seen_non_default_actions.add(action)
                for conflict_action in action_conflicts.get(action, []):
                    if conflict_action in seen_non_default_actions:
                        msg = _('not allowed with argument %s')
                        action_name = _get_action_name(conflict_action)
                        raise ArgumentError(action, msg % action_name)
            if argument_values is not SUPPRESS:
                action(self, namespace, argument_values, option_string)

        def consume_optional(start_index):
            option_tuple = option_string_indices[start_index]
            action, option_string, explicit_arg = option_tuple
            match_argument = self._match_argument
            action_tuples = []
            while True:
                if action is None:
                    extras.append(arg_strings[start_index])
                    return start_index + 1
                if explicit_arg is not None:
                    arg_count = match_argument(action, 'A')
                    chars = self.prefix_chars
                    if arg_count == 0 and option_string[1] not in chars:
                        action_tuples.append((action, [], option_string))
                        char = option_string[0]
                        option_string = char + explicit_arg[0]
                        new_explicit_arg = explicit_arg[1:] or None
                        optionals_map = self._option_string_actions
                        if option_string in optionals_map:
                            action = optionals_map[option_string]
                            explicit_arg = new_explicit_arg
                        else:
                            msg = _('ignored explicit argument %r')
                            raise ArgumentError(action, msg % explicit_arg)
                    elif arg_count == 1:
                        stop = start_index + 1
                        args = [explicit_arg]
                        action_tuples.append((action, args, option_string))
                        break
                    else:
                        msg = _('ignored explicit argument %r')
                        raise ArgumentError(action, msg % explicit_arg)
                else:
                    start = start_index + 1
                    selected_patterns = arg_strings_pattern[start:]
                    arg_count = match_argument(action, selected_patterns)
                    stop = start + arg_count
                    args = arg_strings[start:stop]
                    action_tuples.append((action, args, option_string))
                    break
            assert action_tuples
            for action, args, option_string in action_tuples:
                take_action(action, args, option_string)
            return stop
        positionals = self._get_positional_actions()

        def consume_positionals(start_index):
            match_partial = self._match_arguments_partial
            selected_pattern = arg_strings_pattern[start_index:]
            arg_counts = match_partial(positionals, selected_pattern)
            for action, arg_count in zip(positionals, arg_counts):
                args = arg_strings[start_index:start_index + arg_count]
                start_index += arg_count
                take_action(action, args)
            positionals[:] = positionals[len(arg_counts):]
            return start_index
        extras = []
        start_index = 0
        if option_string_indices:
            max_option_string_index = max(option_string_indices)
        else:
            max_option_string_index = -1
        while start_index <= max_option_string_index:
            next_option_string_index = min([index for index in option_string_indices if index >= start_index])
            if start_index != next_option_string_index:
                positionals_end_index = consume_positionals(start_index)
                if positionals_end_index > start_index:
                    start_index = positionals_end_index
                    continue
                else:
                    start_index = positionals_end_index
            if start_index not in option_string_indices:
                strings = arg_strings[start_index:next_option_string_index]
                extras.extend(strings)
                start_index = next_option_string_index
            start_index = consume_optional(start_index)
        stop_index = consume_positionals(start_index)
        extras.extend(arg_strings[stop_index:])
        if positionals:
            self.error(_('too few arguments'))
        for action in self._actions:
            if action not in seen_actions:
                if action.required:
                    name = _get_action_name(action)
                    self.error(_('argument %s is required') % name)
                elif action.default is not None and isinstance(action.default, basestring) and hasattr(namespace, action.dest) and (action.default is getattr(namespace, action.dest)):
                    setattr(namespace, action.dest, self._get_value(action, action.default))
        for group in self._mutually_exclusive_groups:
            if group.required:
                for action in group._group_actions:
                    if action in seen_non_default_actions:
                        break
                else:
                    names = [_get_action_name(action) for action in group._group_actions if action.help is not SUPPRESS]
                    msg = _('one of the arguments %s is required')
                    self.error(msg % ' '.join(names))
        return (namespace, extras)

    def _read_args_from_files(self, arg_strings):
        new_arg_strings = []
        for arg_string in arg_strings:
            if arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)
            else:
                try:
                    args_file = open(arg_string[1:])
                    try:
                        arg_strings = []
                        for arg_line in args_file.read().splitlines():
                            for arg in self.convert_arg_line_to_args(arg_line):
                                arg_strings.append(arg)
                        arg_strings = self._read_args_from_files(arg_strings)
                        new_arg_strings.extend(arg_strings)
                    finally:
                        args_file.close()
                except IOError:
                    err = _sys.exc_info()[1]
                    self.error(str(err))
        return new_arg_strings

    def convert_arg_line_to_args(self, arg_line):
        return [arg_line]

    def _match_argument(self, action, arg_strings_pattern):
        nargs_pattern = self._get_nargs_pattern(action)
        match = _re.match(nargs_pattern, arg_strings_pattern)
        if match is None:
            nargs_errors = {None: _('expected one argument'), OPTIONAL: _('expected at most one argument'), ONE_OR_MORE: _('expected at least one argument')}
            default = _('expected %s argument(s)') % action.nargs
            msg = nargs_errors.get(action.nargs, default)
            raise ArgumentError(action, msg)
        return len(match.group(1))

    def _match_arguments_partial(self, actions, arg_strings_pattern):
        result = []
        for i in range(len(actions), 0, -1):
            actions_slice = actions[:i]
            pattern = ''.join([self._get_nargs_pattern(action) for action in actions_slice])
            match = _re.match(pattern, arg_strings_pattern)
            if match is not None:
                result.extend([len(string) for string in match.groups()])
                break
        return result

    def _parse_optional(self, arg_string):
        if not arg_string:
            return None
        if not arg_string[0] in self.prefix_chars:
            return None
        if arg_string in self._option_string_actions:
            action = self._option_string_actions[arg_string]
            return (action, arg_string, None)
        if len(arg_string) == 1:
            return None
        if '=' in arg_string:
            option_string, explicit_arg = arg_string.split('=', 1)
            if option_string in self._option_string_actions:
                action = self._option_string_actions[option_string]
                return (action, option_string, explicit_arg)
        option_tuples = self._get_option_tuples(arg_string)
        if len(option_tuples) > 1:
            options = ', '.join([option_string for action, option_string, explicit_arg in option_tuples])
            tup = (arg_string, options)
            self.error(_('ambiguous option: %s could match %s') % tup)
        elif len(option_tuples) == 1:
            option_tuple, = option_tuples
            return option_tuple
        if self._negative_number_matcher.match(arg_string):
            if not self._has_negative_number_optionals:
                return None
        if ' ' in arg_string:
            return None
        return (None, arg_string, None)

    def _get_option_tuples(self, option_string):
        result = []
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            if '=' in option_string:
                option_prefix, explicit_arg = option_string.split('=', 1)
            else:
                option_prefix = option_string
                explicit_arg = None
            for option_string in self._option_string_actions:
                if option_string.startswith(option_prefix):
                    action = self._option_string_actions[option_string]
                    tup = (action, option_string, explicit_arg)
                    result.append(tup)
        elif option_string[0] in chars and option_string[1] not in chars:
            option_prefix = option_string
            explicit_arg = None
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]
            for option_string in self._option_string_actions:
                if option_string == short_option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = (action, option_string, short_explicit_arg)
                    result.append(tup)
                elif option_string.startswith(option_prefix):
                    action = self._option_string_actions[option_string]
                    tup = (action, option_string, explicit_arg)
                    result.append(tup)
        else:
            self.error(_('unexpected option string: %s') % option_string)
        return result

    def _get_nargs_pattern(self, action):
        nargs = action.nargs
        if nargs is None:
            nargs_pattern = '(-*A-*)'
        elif nargs == OPTIONAL:
            nargs_pattern = '(-*A?-*)'
        elif nargs == ZERO_OR_MORE:
            nargs_pattern = '(-*[A-]*)'
        elif nargs == ONE_OR_MORE:
            nargs_pattern = '(-*A[A-]*)'
        elif nargs == REMAINDER:
            nargs_pattern = '([-AO]*)'
        elif nargs == PARSER:
            nargs_pattern = '(-*A[-AO]*)'
        else:
            nargs_pattern = '(-*%s-*)' % '-*'.join('A' * nargs)
        if action.option_strings:
            nargs_pattern = nargs_pattern.replace('-*', '')
            nargs_pattern = nargs_pattern.replace('-', '')
        return nargs_pattern

    def _get_values(self, action, arg_strings):
        if action.nargs not in [PARSER, REMAINDER]:
            arg_strings = [s for s in arg_strings if s != '--']
        if not arg_strings and action.nargs == OPTIONAL:
            if action.option_strings:
                value = action.const
            else:
                value = action.default
            if isinstance(value, basestring):
                value = self._get_value(action, value)
                self._check_value(action, value)
        elif not arg_strings and action.nargs == ZERO_OR_MORE and (not action.option_strings):
            if action.default is not None:
                value = action.default
            else:
                value = arg_strings
            self._check_value(action, value)
        elif len(arg_strings) == 1 and action.nargs in [None, OPTIONAL]:
            arg_string, = arg_strings
            value = self._get_value(action, arg_string)
            self._check_value(action, value)
        elif action.nargs == REMAINDER:
            value = [self._get_value(action, v) for v in arg_strings]
        elif action.nargs == PARSER:
            value = [self._get_value(action, v) for v in arg_strings]
            self._check_value(action, value[0])
        else:
            value = [self._get_value(action, v) for v in arg_strings]
            for v in value:
                self._check_value(action, v)
        return value

    def _get_value(self, action, arg_string):
        type_func = self._registry_get('type', action.type, action.type)
        if not _callable(type_func):
            msg = _('%r is not callable')
            raise ArgumentError(action, msg % type_func)
        try:
            result = type_func(arg_string)
        except ArgumentTypeError:
            name = getattr(action.type, '__name__', repr(action.type))
            msg = str(_sys.exc_info()[1])
            raise ArgumentError(action, msg)
        except (TypeError, ValueError):
            name = getattr(action.type, '__name__', repr(action.type))
            msg = _('invalid %s value: %r')
            raise ArgumentError(action, msg % (name, arg_string))
        return result

    def _check_value(self, action, value):
        if action.choices is not None and value not in action.choices:
            tup = (value, ', '.join(map(repr, action.choices)))
            msg = _('invalid choice: %r (choose from %s)') % tup
            raise ArgumentError(action, msg)

    def format_usage(self):
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        return formatter.format_help()

    def format_help(self):
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()

    def format_version(self):
        import warnings
        warnings.warn('The format_version method is deprecated -- the "version" argument to ArgumentParser is no longer supported.', DeprecationWarning)
        formatter = self._get_formatter()
        formatter.add_text(self.version)
        return formatter.format_help()

    def _get_formatter(self):
        return self.formatter_class(prog=self.prog)

    def print_usage(self, file=None):
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_usage(), file)

    def print_help(self, file=None):
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_help(), file)

    def print_version(self, file=None):
        import warnings
        warnings.warn('The print_version method is deprecated -- the "version" argument to ArgumentParser is no longer supported.', DeprecationWarning)
        self._print_message(self.format_version(), file)

    def _print_message(self, message, file=None):
        if message:
            if file is None:
                file = _sys.stderr
            file.write(message)

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, _sys.stderr)
        _sys.exit(status)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        self.print_usage(_sys.stderr)
        self.exit(2, _('%s: error: %s\n') % (self.prog, message))
import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
class ConfigOpts(abc.Mapping):
    """Config options which may be set on the command line or in config files.

    ConfigOpts is a configuration option manager with APIs for registering
    option schemas, grouping options, parsing option values and retrieving
    the values of options.

    It has built-in support for :oslo.config:option:`config_file` and
    :oslo.config:option:`config_dir` options.

    """
    disallow_names = ('project', 'prog', 'version', 'usage', 'default_config_files', 'default_config_dirs')
    _config_source_opt = ListOpt('config_source', metavar='SOURCE', default=[], help='Lists configuration groups that provide more details for accessing configuration settings from locations other than local files.')

    def __init__(self):
        """Construct a ConfigOpts object."""
        self._opts = {}
        self._groups = {}
        self._deprecated_opts = {}
        self._args = None
        self._oparser = None
        self._namespace = None
        self._mutable_ns = None
        self._mutate_hooks = set([])
        self.__cache = {}
        self.__drivers_cache = {}
        self._config_opts = []
        self._cli_opts = collections.deque()
        self._validate_default_values = False
        self._sources = []
        self._ext_mgr = None
        self._use_env = True
        self._env_driver = _environment.EnvironmentConfigurationSource()
        self.register_opt(self._config_source_opt)

    def _pre_setup(self, project, prog, version, usage, description, epilog, default_config_files, default_config_dirs):
        """Initialize a ConfigCliParser object for option parsing."""
        if prog is None:
            prog = os.path.basename(sys.argv[0])
            if prog.endswith('.py'):
                prog = prog[:-3]
        if default_config_files is None:
            default_config_files = find_config_files(project, prog)
        if default_config_dirs is None:
            default_config_dirs = find_config_dirs(project, prog)
        self._oparser = _CachedArgumentParser(prog=prog, usage=usage, description=description, epilog=epilog)
        if version is not None:
            self._oparser.add_parser_argument(self._oparser, '--version', action='version', version=version)
        return (prog, default_config_files, default_config_dirs)

    @staticmethod
    def _make_config_options(default_config_files, default_config_dirs):
        return [_ConfigFileOpt('config-file', default=default_config_files, metavar='PATH', help='Path to a config file to use. Multiple config files can be specified, with values in later files taking precedence. Defaults to %(default)s. This option must be set from the command-line.'), _ConfigDirOpt('config-dir', metavar='DIR', default=default_config_dirs, help='Path to a config directory to pull `*.conf` files from. This file set is sorted, so as to provide a predictable parse order if individual options are over-ridden. The set is parsed after the file(s) specified via previous --config-file, arguments hence over-ridden options in the directory take precedence. This option must be set from the command-line.')]

    @classmethod
    def _list_options_for_discovery(cls, default_config_files, default_config_dirs):
        """Return options to be used by list_opts() for the sample generator."""
        options = cls._make_config_options(default_config_files, default_config_dirs)
        options.append(cls._config_source_opt)
        return options

    def _setup(self, project, prog, version, usage, default_config_files, default_config_dirs, use_env):
        """Initialize a ConfigOpts object for option parsing."""
        self._config_opts = self._make_config_options(default_config_files, default_config_dirs)
        self.register_cli_opts(self._config_opts)
        self.project = project
        self.prog = prog
        self.version = version
        self.usage = usage
        self.default_config_files = default_config_files
        self.default_config_dirs = default_config_dirs
        self._use_env = use_env

    def __clear_cache(f):

        @functools.wraps(f)
        def __inner(self, *args, **kwargs):
            if kwargs.pop('clear_cache', True):
                result = f(self, *args, **kwargs)
                self.__cache.clear()
                return result
            else:
                return f(self, *args, **kwargs)
        return __inner

    def __clear_drivers_cache(f):

        @functools.wraps(f)
        def __inner(self, *args, **kwargs):
            if kwargs.pop('clear_drivers_cache', True):
                result = f(self, *args, **kwargs)
                self.__drivers_cache.clear()
                return result
            else:
                return f(self, *args, **kwargs)
        return __inner

    def __call__(self, args=None, project=None, prog=None, version=None, usage=None, default_config_files=None, default_config_dirs=None, validate_default_values=False, description=None, epilog=None, use_env=True):
        """Parse command line arguments and config files.

        Calling a ConfigOpts object causes the supplied command line arguments
        and config files to be parsed, causing opt values to be made available
        as attributes of the object.

        The object may be called multiple times, each time causing the previous
        set of values to be overwritten.

        Automatically registers the --config-file option with either a supplied
        list of default config files, or a list from find_config_files().

        If the --config-dir option is set, any *.conf files from this
        directory are pulled in, after all the file(s) specified by the
        --config-file option.

        :param args: command line arguments (defaults to sys.argv[1:])
        :param project: the toplevel project name, used to locate config files
        :param prog: the name of the program (defaults to sys.argv[0]
            basename, without extension .py)
        :param version: the program version (for --version)
        :param usage: a usage string (%prog will be expanded)
        :param description: A description of what the program does
        :param epilog: Text following the argument descriptions
        :param default_config_files: config files to use by default
        :param default_config_dirs: config dirs to use by default
        :param validate_default_values: whether to validate the default values
        :param use_env: If True (the default) look in the environment as one
                        source of option values.
        :raises: SystemExit, ConfigFilesNotFoundError, ConfigFileParseError,
                 ConfigFilesPermissionDeniedError,
                 RequiredOptError, DuplicateOptError
        """
        self.clear()
        self._validate_default_values = validate_default_values
        prog, default_config_files, default_config_dirs = self._pre_setup(project, prog, version, usage, description, epilog, default_config_files, default_config_dirs)
        self._setup(project, prog, version, usage, default_config_files, default_config_dirs, use_env)
        self._namespace = self._parse_cli_opts(args if args is not None else sys.argv[1:])
        if self._namespace._files_not_found:
            raise ConfigFilesNotFoundError(self._namespace._files_not_found)
        if self._namespace._files_permission_denied:
            raise ConfigFilesPermissionDeniedError(self._namespace._files_permission_denied)
        self._load_alternative_sources()
        self._check_required_opts()

    def _load_alternative_sources(self):
        for source_group_name in self.config_source:
            source = self._open_source_from_opt_group(source_group_name)
            if source is not None:
                self._sources.append(source)

    def _open_source_from_opt_group(self, group_name):
        if not self._ext_mgr:
            self._ext_mgr = stevedore.ExtensionManager('oslo.config.driver', invoke_on_load=True)
        self.register_opt(StrOpt('driver', choices=self._ext_mgr.names(), help=_SOURCE_DRIVER_OPTION_HELP), group=group_name)
        try:
            driver_name = self[group_name].driver
        except ConfigFileValueError as err:
            LOG.error('could not load configuration from %r. %s', group_name, err.msg)
            return None
        if driver_name is None:
            LOG.error("could not load configuration from %r, no 'driver' is set.", group_name)
            return None
        LOG.info('loading configuration from %r using %r', group_name, driver_name)
        driver = self._ext_mgr[driver_name].obj
        try:
            return driver.open_source_from_opt_group(self, group_name)
        except Exception as err:
            LOG.error('could not load configuration from %r using %s driver: %s', group_name, driver_name, err)
            return None

    def __getattr__(self, name):
        """Look up an option value and perform string substitution.

        :param name: the opt name (or 'dest', more precisely)
        :returns: the option value (after string substitution) or a GroupAttr
        :raises: ValueError or NoSuchOptError
        """
        try:
            return self._get(name)
        except ValueError:
            raise
        except Exception:
            raise NoSuchOptError(name)

    def __getitem__(self, key):
        """Look up an option value and perform string substitution."""
        return self.__getattr__(key)

    def __contains__(self, key):
        """Return True if key is the name of a registered opt or group."""
        return key in self._opts or key in self._groups

    def __iter__(self):
        """Iterate over all registered opt and group names."""
        for key in itertools.chain(list(self._opts.keys()), list(self._groups.keys())):
            yield key

    def __len__(self):
        """Return the number of options and option groups."""
        return len(self._opts) + len(self._groups)

    def reset(self):
        """Clear the object state and unset overrides and defaults."""
        self._unset_defaults_and_overrides()
        self.clear()

    @__clear_cache
    def clear(self):
        """Reset the state of the object to before options were registered.

        This method removes all registered options and discards the data
        from the command line and configuration files.

        Any subparsers added using the add_cli_subparsers() will also be
        removed as a side-effect of this method.
        """
        self._args = None
        self._oparser = None
        self._namespace = None
        self._mutable_ns = None
        self._validate_default_values = False
        self.unregister_opts(self._config_opts)
        for group in self._groups.values():
            group._clear()

    def _add_cli_opt(self, opt, group):
        if {'opt': opt, 'group': group} in self._cli_opts:
            return
        if opt.positional:
            self._cli_opts.append({'opt': opt, 'group': group})
        else:
            self._cli_opts.appendleft({'opt': opt, 'group': group})

    def _track_deprecated_opts(self, opt, group=None):
        if hasattr(opt, 'deprecated_opts'):
            for dep_opt in opt.deprecated_opts:
                dep_group = dep_opt.group or 'DEFAULT'
                dep_dest = dep_opt.name
                if dep_dest:
                    dep_dest = dep_dest.replace('-', '_')
                if dep_group not in self._deprecated_opts:
                    self._deprecated_opts[dep_group] = {dep_dest: {'opt': opt, 'group': group}}
                else:
                    self._deprecated_opts[dep_group][dep_dest] = {'opt': opt, 'group': group}

    @__clear_cache
    def register_opt(self, opt, group=None, cli=False):
        """Register an option schema.

        Registering an option schema makes any option value which is previously
        or subsequently parsed from the command line or config files available
        as an attribute of this object.

        :param opt: an instance of an Opt sub-class
        :param group: an optional OptGroup object or group name
        :param cli: whether this is a CLI option
        :return: False if the opt was already registered, True otherwise
        :raises: DuplicateOptError
        """
        if group is not None:
            group = self._get_group(group, autocreate=True)
            if cli:
                self._add_cli_opt(opt, group)
            self._track_deprecated_opts(opt, group=group)
            return group._register_opt(opt, cli)
        if group is None:
            if opt.name in self.disallow_names:
                raise ValueError('Name %s was reserved for oslo.config.' % opt.name)
        if cli:
            self._add_cli_opt(opt, None)
        if _is_opt_registered(self._opts, opt):
            return False
        self._opts[opt.dest] = {'opt': opt, 'cli': cli}
        self._track_deprecated_opts(opt)
        return True

    @__clear_cache
    def register_opts(self, opts, group=None):
        """Register multiple option schemas at once."""
        for opt in opts:
            self.register_opt(opt, group, clear_cache=False)

    @__clear_cache
    def register_cli_opt(self, opt, group=None):
        """Register a CLI option schema.

        CLI option schemas must be registered before the command line and
        config files are parsed. This is to ensure that all CLI options are
        shown in --help and option validation works as expected.

        :param opt: an instance of an Opt sub-class
        :param group: an optional OptGroup object or group name
        :return: False if the opt was already registered, True otherwise
        :raises: DuplicateOptError, ArgsAlreadyParsedError
        """
        if self._args is not None:
            raise ArgsAlreadyParsedError('cannot register CLI option')
        return self.register_opt(opt, group, cli=True, clear_cache=False)

    @__clear_cache
    def register_cli_opts(self, opts, group=None):
        """Register multiple CLI option schemas at once."""
        for opt in opts:
            self.register_cli_opt(opt, group, clear_cache=False)

    def register_group(self, group):
        """Register an option group.

        An option group must be registered before options can be registered
        with the group.

        :param group: an OptGroup object
        """
        if group.name in self._groups:
            return
        self._groups[group.name] = copy.copy(group)

    @__clear_cache
    def unregister_opt(self, opt, group=None):
        """Unregister an option.

        :param opt: an Opt object
        :param group: an optional OptGroup object or group name
        :raises: ArgsAlreadyParsedError, NoSuchGroupError
        """
        if self._args is not None:
            raise ArgsAlreadyParsedError('reset before unregistering options')
        remitem = None
        for item in self._cli_opts:
            if item['opt'].dest == opt.dest and (group is None or self._get_group(group).name == item['group'].name):
                remitem = item
                break
        if remitem is not None:
            self._cli_opts.remove(remitem)
        if group is not None:
            self._get_group(group)._unregister_opt(opt)
        elif opt.dest in self._opts:
            del self._opts[opt.dest]

    @__clear_cache
    def unregister_opts(self, opts, group=None):
        """Unregister multiple CLI option schemas at once."""
        for opt in opts:
            self.unregister_opt(opt, group, clear_cache=False)

    def import_opt(self, name, module_str, group=None):
        """Import an option definition from a module.

        Import a module and check that a given option is registered.

        This is intended for use with global configuration objects
        like cfg.CONF where modules commonly register options with
        CONF at module load time. If one module requires an option
        defined by another module it can use this method to explicitly
        declare the dependency.

        :param name: the name/dest of the opt
        :param module_str: the name of a module to import
        :param group: an option OptGroup object or group name
        :raises: NoSuchOptError, NoSuchGroupError
        """
        __import__(module_str)
        self._get_opt_info(name, group)

    def import_group(self, group, module_str):
        """Import an option group from a module.

        Import a module and check that a given option group is registered.

        This is intended for use with global configuration objects
        like cfg.CONF where modules commonly register options with
        CONF at module load time. If one module requires an option group
        defined by another module it can use this method to explicitly
        declare the dependency.

        :param group: an option OptGroup object or group name
        :param module_str: the name of a module to import
        :raises: ImportError, NoSuchGroupError
        """
        __import__(module_str)
        self._get_group(group)

    @__clear_cache
    def set_override(self, name, override, group=None):
        """Override an opt value.

        Override the command line, config file and default values of a
        given option.

        :param name: the name/dest of the opt
        :param override: the override value
        :param group: an option OptGroup object or group name

        :raises: NoSuchOptError, NoSuchGroupError
        """
        opt_info = self._get_opt_info(name, group)
        opt_info['override'] = self._get_enforced_type_value(opt_info['opt'], override)
        opt_info['location'] = LocationInfo(Locations.set_override, _get_caller_detail(3))

    @__clear_cache
    def set_default(self, name, default, group=None):
        """Override an opt's default value.

        Override the default value of given option. A command line or
        config file value will still take precedence over this default.

        :param name: the name/dest of the opt
        :param default: the default value
        :param group: an option OptGroup object or group name

        :raises: NoSuchOptError, NoSuchGroupError
        """
        opt_info = self._get_opt_info(name, group)
        opt_info['default'] = self._get_enforced_type_value(opt_info['opt'], default)
        opt_info['location'] = LocationInfo(Locations.set_default, _get_caller_detail(3))

    def _get_enforced_type_value(self, opt, value):
        if value is None:
            return None
        return self._convert_value(value, opt)

    @__clear_cache
    def clear_override(self, name, group=None):
        """Clear an override an opt value.

        Clear a previously set override of the command line, config file
        and default values of a given option.

        :param name: the name/dest of the opt
        :param group: an option OptGroup object or group name
        :raises: NoSuchOptError, NoSuchGroupError
        """
        opt_info = self._get_opt_info(name, group)
        opt_info.pop('override', None)

    @__clear_cache
    def clear_default(self, name, group=None):
        """Clear an override an opt's default value.

        Clear a previously set override of the default value of given option.

        :param name: the name/dest of the opt
        :param group: an option OptGroup object or group name
        :raises: NoSuchOptError, NoSuchGroupError
        """
        opt_info = self._get_opt_info(name, group)
        opt_info.pop('default', None)

    def _all_opt_infos(self):
        """A generator function for iteration opt infos."""
        for info in self._opts.values():
            yield (info, None)
        for group in self._groups.values():
            for info in group._opts.values():
                yield (info, group)

    def _all_cli_opts(self):
        """A generator function for iterating CLI opts."""
        for item in self._cli_opts:
            yield (item['opt'], item['group'])

    def _unset_defaults_and_overrides(self):
        """Unset any default or override on all options."""
        for info, group in self._all_opt_infos():
            info.pop('default', None)
            info.pop('override', None)

    @property
    def config_dirs(self):
        if self._namespace is None:
            return []
        return self._namespace._config_dirs

    def find_file(self, name):
        """Locate a file located alongside the config files.

        Search for a file with the supplied basename in the directories
        which we have already loaded config files from and other known
        configuration directories.

        The directory, if any, supplied by the config_dir option is
        searched first. Then the config_file option is iterated over
        and each of the base directories of the config_files values
        are searched. Failing both of these, the standard directories
        searched by the module level find_config_files() function is
        used. The first matching file is returned.

        :param name: the filename, for example 'policy.json'
        :returns: the path to a matching file, or None
        """
        if not self._namespace:
            raise NotInitializedError()
        dirs = []
        if self._namespace._config_dirs:
            for directory in self._namespace._config_dirs:
                dirs.append(_fixpath(directory))
        for cf in reversed(self.config_file):
            dirs.append(os.path.dirname(_fixpath(cf)))
        dirs.extend(_get_config_dirs(self.project))
        return _search_dirs(dirs, name)

    def log_opt_values(self, logger, lvl):
        """Log the value of all registered opts.

        It's often useful for an app to log its configuration to a log file at
        startup for debugging. This method dumps to the entire config state to
        the supplied logger at a given log level.

        :param logger: a logging.Logger object
        :param lvl: the log level (for example logging.DEBUG) arg to
                    logger.log()
        """
        logger.log(lvl, '*' * 80)
        logger.log(lvl, 'Configuration options gathered from:')
        logger.log(lvl, 'command line args: %s', self._args)
        logger.log(lvl, 'config files: %s', hasattr(self, 'config_file') and self.config_file or [])
        logger.log(lvl, '=' * 80)

        def _sanitize(opt, value):
            """Obfuscate values of options declared secret."""
            return value if not opt.secret else '*' * 4
        for opt_name in sorted(self._opts):
            opt = self._get_opt_info(opt_name)['opt']
            logger.log(lvl, '%-30s = %s', opt_name, _sanitize(opt, getattr(self, opt_name)))
        for group_name in list(self._groups):
            group_attr = self.GroupAttr(self, self._get_group(group_name))
            for opt_name in sorted(self._groups[group_name]._opts):
                opt = self._get_opt_info(opt_name, group_name)['opt']
                logger.log(lvl, '%-30s = %s', '%s.%s' % (group_name, opt_name), _sanitize(opt, getattr(group_attr, opt_name)))
        logger.log(lvl, '*' * 80)

    def print_usage(self, file=None):
        """Print the usage message for the current program.

        This method is for use after all CLI options are known
        registered using __call__() method. If this method is called
        before the __call__() is invoked, it throws NotInitializedError

        :param file: the File object (if None, output is on sys.stdout)
        :raises: NotInitializedError
        """
        if not self._oparser:
            raise NotInitializedError()
        self._oparser.print_usage(file)

    def print_help(self, file=None):
        """Print the help message for the current program.

        This method is for use after all CLI options are known
        registered using __call__() method. If this method is called
        before the __call__() is invoked, it throws NotInitializedError

        :param file: the File object (if None, output is on sys.stdout)
        :raises: NotInitializedError
        """
        if not self._oparser:
            raise NotInitializedError()
        self._oparser.print_help(file)

    def _get(self, name, group=None, namespace=None):
        if isinstance(group, OptGroup):
            key = (group.name, name)
        else:
            key = (group, name)
        if namespace is None:
            try:
                return self.__cache[key]
            except KeyError:
                pass
        value, loc = self._do_get(name, group, namespace)
        self.__cache[key] = value
        return value

    def _do_get(self, name, group=None, namespace=None):
        """Look up an option value.

        :param name: the opt name (or 'dest', more precisely)
        :param group: an OptGroup
        :param namespace: the namespace object to get the option value from
        :returns: 2-tuple of the option value or a GroupAttr object
                  and LocationInfo or None
        :raises: NoSuchOptError, NoSuchGroupError, ConfigFileValueError,
                 TemplateSubstitutionError
        """
        if group is None and name in self._groups:
            return (self.GroupAttr(self, self._get_group(name)), None)
        info = self._get_opt_info(name, group)
        opt = info['opt']
        if 'location' in info:
            loc = info['location']
        else:
            loc = opt._set_location
        if isinstance(opt, SubCommandOpt):
            return (self.SubCommandAttr(self, group, opt.dest), None)
        if 'override' in info:
            return (self._substitute(info['override']), loc)

        def convert(value):
            return self._convert_value(self._substitute(value, group, namespace), opt)
        group_name = group.name if group else None
        key = (group_name, name)
        env_val = (sources._NoValue, None)
        if self._use_env:
            env_val = self._env_driver.get(group_name, name, opt)
        if opt.mutable and namespace is None:
            namespace = self._mutable_ns
        if namespace is None:
            namespace = self._namespace
        if namespace is not None:
            try:
                alt_loc = None
                try:
                    val, alt_loc = opt._get_from_namespace(namespace, group_name)
                    if val != sources._NoValue and alt_loc.location == Locations.command_line:
                        return (convert(val), alt_loc)
                    if env_val[0] != sources._NoValue:
                        return (convert(env_val[0]), env_val[1])
                    if val != sources._NoValue:
                        return (convert(val), alt_loc)
                except KeyError:
                    alt_loc = LocationInfo(Locations.environment, self._env_driver.get_name(group_name, name))
                    if env_val[0] != sources._NoValue:
                        return (convert(env_val[0]), env_val[1])
            except ValueError as ve:
                message = 'Value for option %s from %s is not valid: %s' % (opt.name, alt_loc, str(ve))
                if alt_loc.location == Locations.user:
                    raise ConfigFileValueError(message)
                raise ConfigSourceValueError(message)
        try:
            return self.__drivers_cache[key]
        except KeyError:
            pass
        for source in self._sources:
            val = source.get(group_name, name, opt)
            if val[0] != sources._NoValue:
                result = (convert(val[0]), val[1])
                self.__drivers_cache[key] = result
                return result
        if 'default' in info:
            return (self._substitute(info['default']), loc)
        if self._validate_default_values:
            if opt.default is not None:
                try:
                    convert(opt.default)
                except ValueError as e:
                    raise ConfigFileValueError('Default value for option %s is not valid: %s' % (opt.name, str(e)))
        if opt.default is not None:
            return (convert(opt.default), loc)
        return (None, None)

    def _substitute(self, value, group=None, namespace=None):
        """Perform string template substitution.

        Substitute any template variables (for example $foo, ${bar}) in
        the supplied string value(s) with opt values.

        :param value: the string value, or list of string values
        :param group: the group that retrieves the option value from
        :param namespace: the namespace object that retrieves the option
                          value from
        :returns: the substituted string(s)
        """
        if isinstance(value, list):
            return [self._substitute(i, group=group, namespace=namespace) for i in value]
        elif isinstance(value, str):
            if '\\$' in value:
                value = value.replace('\\$', '$$')
            tmpl = self.Template(value)
            ret = tmpl.safe_substitute(self.StrSubWrapper(self, group=group, namespace=namespace))
            return ret
        elif isinstance(value, dict):
            return {self._substitute(key, group=group, namespace=namespace): self._substitute(val, group=group, namespace=namespace) for key, val in value.items()}
        else:
            return value

    class Template(string.Template):
        idpattern = '[_a-z][\\._a-z0-9]*'

    def _convert_value(self, value, opt):
        """Perform value type conversion.

        Converts values using option's type. Handles cases when value is
        actually a list of values (for example for multi opts).

        :param value: the string value, or list of string values
        :param opt: option definition (instance of Opt class or its subclasses)
        :returns: converted value
        """
        if opt.multi:
            return [opt.type(v) for v in value]
        else:
            return opt.type(value)

    def _get_group(self, group_or_name, autocreate=False):
        """Looks up a OptGroup object.

        Helper function to return an OptGroup given a parameter which can
        either be the group's name or an OptGroup object.

        The OptGroup object returned is from the internal dict of OptGroup
        objects, which will be a copy of any OptGroup object that users of
        the API have access to.

        If autocreate is True, the group will be created if it's not found. If
        group is an instance of OptGroup, that same instance will be
        registered, otherwise a new instance of OptGroup will be created.

        :param group_or_name: the group's name or the OptGroup object itself
        :param autocreate: whether to auto-create the group if it's not found
        :raises: NoSuchGroupError
        """
        group = group_or_name if isinstance(group_or_name, OptGroup) else None
        group_name = group.name if group else group_or_name
        if group_name not in self._groups:
            if not autocreate:
                raise NoSuchGroupError(group_name)
            self.register_group(group or OptGroup(name=group_name))
        return self._groups[group_name]

    def _find_deprecated_opts(self, opt_name, group=None):
        real_opt_name = None
        real_group_name = None
        group_name = group or 'DEFAULT'
        if hasattr(group_name, 'name'):
            group_name = group_name.name
        dep_group = self._deprecated_opts.get(group_name)
        if dep_group:
            real_opt_dict = dep_group.get(opt_name)
            if real_opt_dict:
                real_opt_name = real_opt_dict['opt'].name
                if real_opt_dict['group']:
                    real_group_name = real_opt_dict['group'].name
        return (real_opt_name, real_group_name)

    def _get_opt_info(self, opt_name, group=None):
        """Return the (opt, override, default) dict for an opt.

        :param opt_name: an opt name/dest
        :param group: an optional group name or OptGroup object
        :raises: NoSuchOptError, NoSuchGroupError
        """
        if group is None:
            opts = self._opts
        else:
            group = self._get_group(group)
            opts = group._opts
        if opt_name not in opts:
            real_opt_name, real_group_name = self._find_deprecated_opts(opt_name, group=group)
            if not real_opt_name:
                raise NoSuchOptError(opt_name, group)
            log_real_group_name = real_group_name or 'DEFAULT'
            dep_message = 'Config option %(dep_group)s.%(dep_option)s  is deprecated. Use option %(group)s.%(option)s instead.'
            LOG.warning(dep_message, {'dep_option': opt_name, 'dep_group': group, 'option': real_opt_name, 'group': log_real_group_name})
            opt_name = real_opt_name
            if real_group_name:
                group = self._get_group(real_group_name)
                opts = group._opts
        return opts[opt_name]

    def _check_required_opts(self, namespace=None):
        """Check that all opts marked as required have values specified.

        :param namespace: the namespace object be checked the required options
        :raises: RequiredOptError
        """
        for info, group in self._all_opt_infos():
            opt = info['opt']
            if opt.required:
                if 'default' in info or 'override' in info:
                    continue
                if self._get(opt.dest, group, namespace) is None:
                    raise RequiredOptError(opt.name, group)

    def _parse_cli_opts(self, args):
        """Parse command line options.

        Initializes the command line option parser and parses the supplied
        command line arguments.

        :param args: the command line arguments
        :returns: a _Namespace object containing the parsed option values
        :raises: SystemExit, DuplicateOptError
                 ConfigFileParseError, ConfigFileValueError

        """
        self._args = args
        for opt, group in self._all_cli_opts():
            opt._add_to_cli(self._oparser, group)
        return self._parse_config_files()

    def _parse_config_files(self):
        """Parse configure files options.

        :raises: SystemExit, ConfigFilesNotFoundError, ConfigFileParseError,
                 ConfigFilesPermissionDeniedError,
                 RequiredOptError, DuplicateOptError
        """
        namespace = _Namespace(self)
        for arg in self._args:
            if arg == '--config-file' or arg.startswith('--config-file='):
                break
        else:
            for config_file in self.default_config_files:
                ConfigParser._parse_file(config_file, namespace)
        for arg in self._args:
            if arg == '--config-dir' or arg.startswith('--config-dir='):
                break
        else:
            for config_dir in self.default_config_dirs:
                if not os.path.exists(config_dir):
                    continue
                config_dir_glob = os.path.join(config_dir, '*.conf')
                for config_file in sorted(glob.glob(config_dir_glob)):
                    ConfigParser._parse_file(config_file, namespace)
        self._oparser.parse_args(self._args, namespace)
        self._validate_cli_options(namespace)
        return namespace

    def _validate_cli_options(self, namespace):
        for opt, group in sorted(self._all_cli_opts(), key=lambda x: x[0].name):
            group_name = group.name if group else None
            try:
                value, loc = opt._get_from_namespace(namespace, group_name)
            except KeyError:
                continue
            value = self._substitute(value, group=group, namespace=namespace)
            try:
                self._convert_value(value, opt)
            except ValueError:
                sys.stderr.write('argument --%s: Invalid %s value: %s\n' % (opt.dest, repr(opt.type), value))
                raise SystemExit

    def _reload_config_files(self):
        namespace = self._parse_config_files()
        if namespace._files_not_found:
            raise ConfigFilesNotFoundError(namespace._files_not_found)
        if namespace._files_permission_denied:
            raise ConfigFilesPermissionDeniedError(namespace._files_permission_denied)
        self._check_required_opts(namespace)
        return namespace

    @__clear_cache
    @__clear_drivers_cache
    def reload_config_files(self):
        """Reload configure files and parse all options

        :return: False if reload configure files failed or else return True
        """
        try:
            namespace = self._reload_config_files()
        except SystemExit as exc:
            LOG.warning('Caught SystemExit while reloading configure files with exit code: %d', exc.code)
            return False
        except Error as err:
            LOG.warning('Caught Error while reloading configure files: %s', err)
            return False
        else:
            self._namespace = namespace
            return True

    def register_mutate_hook(self, hook):
        """Registers a hook to be called by mutate_config_files.

        :param hook: a function accepting this ConfigOpts object and a dict of
                     config mutations, as returned by mutate_config_files.
        :return: None
        """
        self._mutate_hooks.add(hook)

    def mutate_config_files(self):
        """Reload configure files and parse all options.

        Only options marked as 'mutable' will appear to change.

        Hooks are called in a NON-DETERMINISTIC ORDER. Do not expect hooks to
        be called in the same order as they were added.

        :return: {(None or 'group', 'optname'): (old_value, new_value), ... }
        :raises: Error if reloading fails
        """
        self.__cache.clear()
        old_mutate_ns = self._mutable_ns or self._namespace
        self._mutable_ns = self._reload_config_files()
        self._warn_immutability()
        fresh = self._diff_ns(old_mutate_ns, self._mutable_ns)

        def key_fn(item):
            groupname, optname = item[0]
            return item[0] if groupname else ('\t', optname)
        sorted_fresh = sorted(fresh.items(), key=key_fn)
        for (groupname, optname), (old, new) in sorted_fresh:
            groupname = groupname if groupname else 'DEFAULT'
            LOG.info('Option %(group)s.%(option)s changed from [%(old_val)s] to [%(new_val)s]', {'group': groupname, 'option': optname, 'old_val': old, 'new_val': new})
        for hook in self._mutate_hooks:
            hook(self, fresh)
        return fresh

    def _warn_immutability(self):
        """Check immutable opts have not changed.

        _do_get won't return the new values but presumably someone changed the
        config file expecting them to change so we should warn them they won't.
        """
        for info, group in self._all_opt_infos():
            opt = info['opt']
            if opt.mutable:
                continue
            groupname = group.name if group else 'DEFAULT'
            try:
                old, _ = opt._get_from_namespace(self._namespace, groupname)
            except KeyError:
                old = None
            try:
                new, _ = opt._get_from_namespace(self._mutable_ns, groupname)
            except KeyError:
                new = None
            if old != new:
                LOG.warning('Ignoring change to immutable option %(group)s.%(option)s', {'group': groupname, 'option': opt.name})

    def _diff_ns(self, old_ns, new_ns):
        """Compare mutable option values between two namespaces.

        This can be used to only reconfigure stateful sessions when necessary.

        :return {(None or 'group', 'optname'): (old_value, new_value), ... }
        """
        diff = {}
        for info, group in self._all_opt_infos():
            opt = info['opt']
            if not opt.mutable:
                continue
            groupname = group.name if group else None
            try:
                old, _ = opt._get_from_namespace(old_ns, groupname)
            except KeyError:
                old = None
            try:
                new, _ = opt._get_from_namespace(new_ns, groupname)
            except KeyError:
                new = None
            if old != new:
                diff[groupname, opt.name] = (old, new)
        return diff

    def list_all_sections(self):
        """List all sections from the configuration.

        Returns a sorted list of all section names found in the
        configuration files, whether declared beforehand or not.
        """
        s = set([])
        if self._mutable_ns:
            s |= set(self._mutable_ns._sections())
        if self._namespace:
            s |= set(self._namespace._sections())
        return sorted(s)

    def get_location(self, name, group=None):
        """Return the location where the option is being set.

        :param name: The name of the option.
        :type name: str
        :param group: The name of the group of the option. Defaults to
                      ``'DEFAULT'``.
        :type group: str
        :return: LocationInfo

        .. seealso::

           :doc:`/reference/locations`

        .. versionadded:: 5.3.0
        """
        opt_group = OptGroup(group) if group is not None else None
        value, loc = self._do_get(name, opt_group, None)
        return loc

    class GroupAttr(abc.Mapping):
        """Helper class.

        Represents the option values of a group as a mapping and attributes.
        """

        def __init__(self, conf, group):
            """Construct a GroupAttr object.

            :param conf: a ConfigOpts object
            :param group: an OptGroup object
            """
            self._conf = conf
            self._group = group

        def __getattr__(self, name):
            """Look up an option value and perform template substitution."""
            return self._conf._get(name, self._group)

        def __getitem__(self, key):
            """Look up an option value and perform string substitution."""
            return self.__getattr__(key)

        def __contains__(self, key):
            """Return True if key is the name of a registered opt or group."""
            return key in self._group._opts

        def __iter__(self):
            """Iterate over all registered opt and group names."""
            for key in self._group._opts.keys():
                yield key

        def __len__(self):
            """Return the number of options and option groups."""
            return len(self._group._opts)

    class SubCommandAttr:
        """Helper class.

        Represents the name and arguments of an argparse sub-parser.
        """

        def __init__(self, conf, group, dest):
            """Construct a SubCommandAttr object.

            :param conf: a ConfigOpts object
            :param group: an OptGroup object
            :param dest: the name of the sub-parser
            """
            self._conf = conf
            self._group = group
            self._dest = dest

        def __getattr__(self, name):
            """Look up a sub-parser name or argument value."""
            if name == 'name':
                name = self._dest
                if self._group is not None:
                    name = self._group.name + '_' + name
                return getattr(self._conf._namespace, name)
            if name in self._conf:
                raise DuplicateOptError(name)
            try:
                return getattr(self._conf._namespace, name)
            except AttributeError:
                raise NoSuchOptError(name)

    class StrSubWrapper:
        """Helper class.

        Exposes opt values as a dict for string substitution.
        """

        def __init__(self, conf, group=None, namespace=None):
            """Construct a StrSubWrapper object.

            :param conf: a ConfigOpts object
            :param group: an OptGroup object
            :param namespace: the namespace object that retrieves the option
                              value from
            """
            self.conf = conf
            self.group = group
            self.namespace = namespace

        def __getitem__(self, key):
            """Look up an opt value from the ConfigOpts object.

            :param key: an opt name
            :returns: an opt value
            """
            try:
                group_name, option = key.split('.', 1)
            except ValueError:
                group = self.group
                option = key
            else:
                group = OptGroup(name=group_name)
            try:
                value = self.conf._get(option, group=group, namespace=self.namespace)
            except NoSuchOptError:
                value = self.conf._get(key, namespace=self.namespace)
            if isinstance(value, self.conf.GroupAttr):
                raise TemplateSubstitutionError('substituting group %s not supported' % key)
            if value is None:
                return ''
            return value
import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources
class AbstractInstallCommand(Command):
    default_interactive = 1
    default_sysconfigs = [(False, '/etc/paste/sysconfig.py'), (False, '/usr/local/etc/paste/sysconfig.py'), (True, 'paste.script.default_sysconfig')]
    if os.environ.get('HOME'):
        default_sysconfigs.insert(0, (False, os.path.join(os.environ['HOME'], '.paste', 'config', 'sysconfig.py')))
    if os.environ.get('PASTE_SYSCONFIG'):
        default_sysconfigs.insert(0, (False, os.environ['PASTE_SYSCONFIG']))

    def run(self, args):
        self.sysconfigs = self.default_sysconfigs
        new_args = []
        while args:
            if args[0].startswith('--no-default-sysconfig'):
                self.sysconfigs = []
                args.pop(0)
                continue
            if args[0].startswith('--sysconfig='):
                self.sysconfigs.insert(0, (True, args.pop(0)[len('--sysconfig='):]))
                continue
            if args[0] == '--sysconfig':
                args.pop(0)
                if not args:
                    raise BadCommand('You gave --sysconfig as the last argument without a value')
                self.sysconfigs.insert(0, (True, args.pop(0)))
                continue
            new_args.append(args.pop(0))
        self.load_sysconfigs()
        return super(AbstractInstallCommand, self).run(new_args)

    def standard_parser(cls, **kw):
        parser = super(AbstractInstallCommand, cls).standard_parser(**kw)
        parser.add_option('--sysconfig', action='append', dest='sysconfigs', help='System configuration file')
        parser.add_option('--no-default-sysconfig', action='store_true', dest='no_default_sysconfig', help="Don't load the default sysconfig files")
        parser.add_option('--easy-install', action='append', dest='easy_install_op', metavar='OP', help='An option to add if invoking easy_install (like --easy-install=exclude-scripts)')
        parser.add_option('--no-install', action='store_true', dest='no_install', help="Don't try to install the package (it must already be installed)")
        parser.add_option('-f', '--find-links', action='append', dest='easy_install_find_links', metavar='URL', help='Passed through to easy_install')
        return parser
    standard_parser = classmethod(standard_parser)

    def load_sysconfigs(self):
        configs = self.sysconfigs[:]
        configs.reverse()
        self.sysconfig_modules = []
        for index, (explicit, name) in enumerate(configs):
            if name.endswith('.py'):
                if not os.path.exists(name):
                    if explicit:
                        raise BadCommand('sysconfig file %s does not exist' % name)
                    else:
                        continue
                globs = {}
                exec(compile(open(name).read(), name, 'exec'), globs)
                mod = types.ModuleType('__sysconfig_%i__' % index)
                for name, value in globs.items():
                    setattr(mod, name, value)
                mod.__file__ = name
            else:
                try:
                    mod = import_string.simple_import(name)
                except ImportError:
                    if explicit:
                        raise
                    else:
                        continue
            mod.paste_command = self
            self.sysconfig_modules.insert(0, mod)
        parser = self.parser
        self.call_sysconfig_functions('add_custom_options', parser)

    def get_sysconfig_option(self, name, default=None):
        """
        Return the value of the given option in the first sysconfig
        module in which it is found, or ``default`` (None) if not
        found in any.
        """
        for mod in self.sysconfig_modules:
            if hasattr(mod, name):
                return getattr(mod, name)
        return default

    def get_sysconfig_options(self, name):
        """
        Return the option value for the given name in all the
        sysconfig modules in which is is found (``[]`` if none).
        """
        return [getattr(mod, name) for mod in self.sysconfig_modules if hasattr(mod, name)]

    def call_sysconfig_function(self, name, *args, **kw):
        """
        Call the specified function in the first sysconfig module it
        is defined in.  ``NameError`` if no function is found.
        """
        val = self.get_sysconfig_option(name)
        if val is None:
            raise NameError('Method %s not found in any sysconfig module' % name)
        return val(*args, **kw)

    def call_sysconfig_functions(self, name, *args, **kw):
        """
        Call all the named functions in the sysconfig modules,
        returning a list of the return values.
        """
        return [method(*args, **kw) for method in self.get_sysconfig_options(name)]

    def sysconfig_install_vars(self, installer):
        """
        Return the folded results of calling the
        ``install_variables()`` functions.
        """
        result = {}
        all_vars = self.call_sysconfig_functions('install_variables', installer)
        all_vars.reverse()
        for vardict in all_vars:
            result.update(vardict)
        return result

    def get_distribution(self, req):
        """
        This gets a distribution object, and installs the distribution
        if required.
        """
        try:
            dist = pkg_resources.get_distribution(req)
            if self.verbose:
                print('Distribution already installed:')
                print(' ', dist, 'from', dist.location)
            return dist
        except pkg_resources.DistributionNotFound:
            if self.options.no_install:
                print("Because --no-install was given, we won't try to install the package %s" % req)
                raise
            options = ['-v', '-m']
            for op in self.options.easy_install_op or []:
                if not op.startswith('-'):
                    op = '--' + op
                options.append(op)
            for op in self.options.easy_install_find_links or []:
                options.append('--find-links=%s' % op)
            if self.simulate:
                raise BadCommand('Must install %s, but in simulation mode' % req)
            print('Must install %s' % req)
            from setuptools.command import easy_install
            from setuptools import setup
            setup(script_args=['-q', 'easy_install'] + options + [req])
            return pkg_resources.get_distribution(req)

    def get_installer(self, distro, ep_group, ep_name):
        if hasattr(distro, 'load_entry_point'):
            installer_class = distro.load_entry_point('paste.app_install', ep_name)
        else:
            eps = [ep for ep in distro.entry_points if ep.group == 'paste.app_install' and ep.name == ep_name]
            installer_class = eps[0].load()
        installer = installer_class(distro, ep_group, ep_name)
        return installer
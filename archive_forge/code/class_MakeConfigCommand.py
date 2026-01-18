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
class MakeConfigCommand(AbstractInstallCommand):
    default_verbosity = 1
    max_args = None
    min_args = 1
    summary = 'Install a package and create a fresh config file/directory'
    usage = 'PACKAGE_NAME [CONFIG_FILE] [VAR=VALUE]'
    description = '    Note: this is an experimental command, and it will probably change\n    in several ways by the next release.\n\n    make-config is part of a two-phase installation process (the\n    second phase is setup-app).  make-config installs the package\n    (using easy_install) and asks it to create a bare configuration\n    file or directory (possibly filling in defaults from the extra\n    variables you give).\n    '
    parser = AbstractInstallCommand.standard_parser(simulate=True, quiet=True, no_interactive=True)
    parser.add_option('--info', action='store_true', dest='show_info', help='Show information on the package (after installing it), but do not write a config.')
    parser.add_option('--name', action='store', dest='ep_name', help='The name of the application contained in the distribution (default "main")')
    parser.add_option('--entry-group', action='store', dest='ep_group', default='paste.app_factory', help='The entry point group to install (i.e., the kind of application; default paste.app_factory')
    parser.add_option('--edit', action='store_true', dest='edit', help='Edit the configuration file after generating it (using $EDITOR)')
    parser.add_option('--setup', action='store_true', dest='run_setup', help='Run setup-app immediately after generating (and possibly editing) the configuration file')

    def command(self):
        self.requirement = self.args[0]
        if '#' in self.requirement:
            if self.options.ep_name is not None:
                raise BadCommand('You may not give both --name and a requirement with #name')
            self.requirement, self.options.ep_name = self.requirement.split('#', 1)
        if not self.options.ep_name:
            self.options.ep_name = 'main'
        self.distro = self.get_distribution(self.requirement)
        self.installer = self.get_installer(self.distro, self.options.ep_group, self.options.ep_name)
        if self.options.show_info:
            if len(self.args) > 1:
                raise BadCommand('With --info you can only give one argument')
            return self.show_info()
        if len(self.args) < 2:
            options = filter(None, self.call_sysconfig_functions('default_config_filename', self.installer))
            if not options:
                raise BadCommand('You must give a configuration filename')
            self.config_file = options[0]
        else:
            self.config_file = self.args[1]
        self.check_config_file()
        self.project_name = self.distro.project_name
        self.vars = self.sysconfig_install_vars(self.installer)
        self.vars.update(self.parse_vars(self.args[2:]))
        self.vars['project_name'] = self.project_name
        self.vars['requirement'] = self.requirement
        self.vars['ep_name'] = self.options.ep_name
        self.vars['ep_group'] = self.options.ep_group
        self.vars.setdefault('app_name', self.project_name.lower())
        self.vars.setdefault('app_instance_uuid', uuid.uuid4())
        self.vars.setdefault('app_instance_secret', secret.secret_string())
        if self.verbose > 1:
            print_vars = sorted(self.vars.items())
            print('Variables for installation:')
            for name, value in print_vars:
                print('  %s: %r' % (name, value))
        self.installer.write_config(self, self.config_file, self.vars)
        edit_success = True
        if self.options.edit:
            edit_success = self.run_editor()
        setup_configs = self.installer.editable_config_files(self.config_file)
        setup_config = setup_configs[0]
        if self.options.run_setup:
            if not edit_success:
                print('Config-file editing was not successful.')
                if self.ask('Run setup-app anyway?', default=False):
                    self.run_setup(setup_config)
            else:
                self.run_setup(setup_config)
        else:
            filenames = self.installer.editable_config_files(self.config_file)
            assert not isinstance(filenames, str), 'editable_config_files returned a string, not a list'
            if not filenames and filenames is not None:
                print('No config files need editing')
            else:
                print('Now you should edit the config files')
                if filenames:
                    for fn in filenames:
                        print('  %s' % fn)

    def show_info(self):
        text = self.installer.description(None)
        print(text)

    def check_config_file(self):
        if self.installer.expect_config_directory is None:
            return
        fn = self.config_file
        if self.installer.expect_config_directory:
            if os.path.splitext(fn)[1]:
                raise BadCommand('The CONFIG_FILE argument %r looks like a filename, and a directory name is expected' % fn)
        elif fn.endswith('/') or not os.path.splitext(fn):
            raise BadCommand('The CONFIG_FILE argument %r looks like a directory name and a filename is expected' % fn)

    def run_setup(self, filename):
        run_command(['setup-app', filename])

    def run_editor(self):
        filenames = self.installer.editable_config_files(self.config_file)
        if filenames is None:
            print('Warning: the config file is not known (--edit ignored)')
            return False
        if not filenames:
            print('Warning: no config files need editing (--edit ignored)')
            return True
        if len(filenames) > 1:
            print('Warning: there is more than one editable config file (--edit ignored)')
            return False
        if not os.environ.get('EDITOR'):
            print('Error: you must set $EDITOR if using --edit')
            return False
        if self.verbose:
            print('%s %s' % (os.environ['EDITOR'], filenames[0]))
        retval = os.system('$EDITOR %s' % filenames[0])
        if retval:
            print('Warning: editor %s returned with error code %i' % (os.environ['EDITOR'], retval))
            return False
        return True
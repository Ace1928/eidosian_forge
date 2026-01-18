import getpass
import logging
import sys
import traceback
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from oslo_utils import importutils
from oslo_utils import strutils
from osc_lib.cli import client_config as cloud_config
from osc_lib import clientmanager
from osc_lib.command import timing
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import logs
from osc_lib import utils
from osc_lib import version
class OpenStackShell(app.App):
    CONSOLE_MESSAGE_FORMAT = '%(levelname)s: %(name)s %(message)s'
    log = logging.getLogger(__name__)
    timing_data = []

    def __init__(self, description=None, version=None, command_manager=None, stdin=None, stdout=None, stderr=None, interactive_app_factory=None, deferred_help=False):
        command.Command.auth_required = True
        help.HelpCommand.auth_required = False
        complete.CompleteCommand.auth_required = False
        self.DEFAULT_DEBUG_VALUE = None
        self.DEFAULT_DEBUG_HELP = 'Set debug logging and traceback on errors.'
        if not command_manager:
            cm = commandmanager.CommandManager('openstack.cli')
        else:
            cm = command_manager
        super(OpenStackShell, self).__init__(description=__doc__.strip(), version=version, command_manager=cm, deferred_help=True)
        self.dump_stack_trace = True
        self.api_version = None
        self.client_manager = None
        self.command_options = None
        self.do_profile = False

    def configure_logging(self):
        """Configure logging for the app."""
        self.log_configurator = logs.LogConfigurator(self.options)
        self.dump_stack_trace = self.log_configurator.dump_trace

    def run(self, argv):
        ret_val = 1
        self.command_options = argv
        try:
            ret_val = super(OpenStackShell, self).run(argv)
            return ret_val
        except Exception as e:
            if not logging.getLogger('').handlers:
                logging.basicConfig()
            if self.dump_stack_trace:
                self.log.error(traceback.format_exc())
            else:
                self.log.error('Exception raised: ' + str(e))
            return ret_val
        finally:
            self.log.info('END return value: %s', ret_val)

    def init_profile(self):
        self.do_profile = osprofiler_profiler and self.options.profile
        if self.do_profile:
            osprofiler_profiler.init(self.options.profile)

    def close_profile(self):
        if self.do_profile:
            profiler = osprofiler_profiler.get()
            trace_id = profiler.get_base_id()
            short_id = profiler.get_shorten_id(trace_id)
            self.log.warning('Trace ID: %s' % trace_id)
            self.log.warning('Short trace ID for OpenTracing-based drivers: %s' % short_id)
            self.log.warning('Display trace data with command:\nosprofiler trace show --html %s ' % trace_id)

    def run_subcommand(self, argv):
        self.init_profile()
        try:
            ret_value = super(OpenStackShell, self).run_subcommand(argv)
        finally:
            self.close_profile()
        return ret_value

    def interact(self):
        self.init_profile()
        try:
            ret_value = super(OpenStackShell, self).interact()
        finally:
            self.close_profile()
        return ret_value

    def build_option_parser(self, description, version):
        parser = super(OpenStackShell, self).build_option_parser(description, version)
        parser.add_argument('--os-cloud', metavar='<cloud-config-name>', dest='cloud', default=utils.env('OS_CLOUD'), help=_('Cloud name in clouds.yaml (Env: OS_CLOUD)'))
        parser.add_argument('--os-region-name', metavar='<auth-region-name>', dest='region_name', default=utils.env('OS_REGION_NAME'), help=_('Authentication region name (Env: OS_REGION_NAME)'))
        parser.add_argument('--os-cacert', metavar='<ca-bundle-file>', dest='cacert', default=utils.env('OS_CACERT', default=None), help=_('CA certificate bundle file (Env: OS_CACERT)'))
        parser.add_argument('--os-cert', metavar='<certificate-file>', dest='cert', default=utils.env('OS_CERT'), help=_('Client certificate bundle file (Env: OS_CERT)'))
        parser.add_argument('--os-key', metavar='<key-file>', dest='key', default=utils.env('OS_KEY'), help=_('Client certificate key file (Env: OS_KEY)'))
        verify_group = parser.add_mutually_exclusive_group()
        verify_group.add_argument('--verify', action='store_true', default=None, help=_('Verify server certificate (default)'))
        verify_group.add_argument('--insecure', action='store_true', default=None, help=_('Disable server certificate verification'))
        parser.add_argument('--os-default-domain', metavar='<auth-domain>', dest='default_domain', default=utils.env('OS_DEFAULT_DOMAIN', default=DEFAULT_DOMAIN), help=_('Default domain ID, default=%s. (Env: OS_DEFAULT_DOMAIN)') % DEFAULT_DOMAIN)
        parser.add_argument('--os-interface', metavar='<interface>', dest='interface', choices=['admin', 'public', 'internal'], default=utils.env('OS_INTERFACE'), help=_('Select an interface type. Valid interface types: [admin, public, internal]. default=%s, (Env: OS_INTERFACE)') % DEFAULT_INTERFACE)
        parser.add_argument('--os-service-provider', metavar='<service_provider>', dest='service_provider', default=utils.env('OS_SERVICE_PROVIDER'), help=_('Authenticate with and perform the command on a service provider using Keystone-to-keystone federation. Must also specify the remote project option.'))
        remote_project_group = parser.add_mutually_exclusive_group()
        remote_project_group.add_argument('--os-remote-project-name', metavar='<remote_project_name>', dest='remote_project_name', default=utils.env('OS_REMOTE_PROJECT_NAME'), help=_('Project name when authenticating to a service provider if using Keystone-to-Keystone federation.'))
        remote_project_group.add_argument('--os-remote-project-id', metavar='<remote_project_id>', dest='remote_project_id', default=utils.env('OS_REMOTE_PROJECT_ID'), help=_('Project ID when authenticating to a service provider if using Keystone-to-Keystone federation.'))
        remote_project_domain_group = parser.add_mutually_exclusive_group()
        remote_project_domain_group.add_argument('--os-remote-project-domain-name', metavar='<remote_project_domain_name>', dest='remote_project_domain_name', default=utils.env('OS_REMOTE_PROJECT_DOMAIN_NAME'), help=_('Domain name of the project when authenticating to a service provider if using Keystone-to-Keystone federation.'))
        remote_project_domain_group.add_argument('--os-remote-project-domain-id', metavar='<remote_project_domain_id>', dest='remote_project_domain_id', default=utils.env('OS_REMOTE_PROJECT_DOMAIN_ID'), help=_('Domain ID of the project when authenticating to a service provider if using Keystone-to-Keystone federation.'))
        parser.add_argument('--timing', default=False, action='store_true', help=_('Print API call timing info'))
        parser.add_argument('--os-beta-command', action='store_true', help=_('Enable beta commands which are subject to change'))
        if osprofiler_profiler:
            parser.add_argument('--os-profile', metavar='hmac-key', dest='profile', default=utils.env('OS_PROFILE'), help=_('HMAC key for encrypting profiling context data'))
        return parser
    '\n    Break up initialize_app() so that overriding it in a subclass does not\n    require duplicating a lot of the method\n\n    * super()\n    * _final_defaults()\n    * OpenStackConfig\n    * get_one\n    * _load_plugins()\n    * _load_commands()\n    * ClientManager\n\n    '

    def _final_defaults(self):
        self._auth_type = None
        project_id = getattr(self.options, 'project_id', None)
        project_name = getattr(self.options, 'project_name', None)
        tenant_id = getattr(self.options, 'tenant_id', None)
        tenant_name = getattr(self.options, 'tenant_name', None)
        if project_id and (not tenant_id):
            self.options.tenant_id = project_id
        if project_name and (not tenant_name):
            self.options.tenant_name = project_name
        if tenant_id and (not project_id):
            self.options.project_id = tenant_id
        if tenant_name and (not project_name):
            self.options.project_name = tenant_name
        self.default_domain = self.options.default_domain

    def _load_plugins(self):
        """Load plugins via stevedore

        osc-lib has no opinion on what plugins should be loaded
        """
        pass

    def _load_commands(self):
        """Load commands via cliff/stevedore

        osc-lib has no opinion on what commands should be loaded
        """
        pass

    def initialize_app(self, argv):
        """Global app init bits:

        * set up API versions
        * validate authentication info
        * authenticate against Identity if requested
        """
        super(OpenStackShell, self).initialize_app(argv)
        self.log.info('START with options: %s', strutils.mask_password(' '.join(self.command_options)))
        self.log.debug('options: %s', strutils.mask_password(self.options))
        self._final_defaults()
        try:
            self.cloud_config = cloud_config.OSC_Config(pw_func=prompt_for_password, override_defaults={'interface': DEFAULT_INTERFACE, 'auth_type': self._auth_type})
        except (IOError, OSError) as e:
            self.log.critical('Could not read clouds.yaml configuration file')
            self.print_help_if_requested()
            raise e
        if not self.options.debug:
            self.options.debug = None
        self.cloud = self.cloud_config.get_one(cloud=self.options.cloud, argparse=self.options, validate=False)
        self.log_configurator.configure(self.cloud)
        self.dump_stack_trace = self.log_configurator.dump_trace
        self.log.debug('defaults: %s', self.cloud_config.defaults)
        self.log.debug('cloud cfg: %s', strutils.mask_password(self.cloud.config))
        self._load_plugins()
        self._load_commands()
        self.print_help_if_requested()
        self.client_manager = clientmanager.ClientManager(cli_options=self.cloud, api_version=self.api_version, pw_func=prompt_for_password)

    def prepare_to_run_command(self, cmd):
        """Set up auth and API versions"""
        self.log.info('command: %s -> %s.%s (auth=%s)', getattr(cmd, 'cmd_name', '<none>'), cmd.__class__.__module__, cmd.__class__.__name__, cmd.auth_required)
        validate = cmd.auth_required
        self.client_manager._auth_required = cmd.auth_required
        self.cloud = self.cloud_config.get_one(cloud=self.options.cloud, argparse=self.options, validate=validate, app_name=self.client_manager._app_name, app_version=self.client_manager._app_version, additional_user_agent=[('osc-lib', version.version_string)])
        self.client_manager._cli_options = self.cloud
        if cmd.auth_required:
            self.client_manager.setup_auth()
            if hasattr(cmd, 'required_scope') and cmd.required_scope:
                self.client_manager.validate_scope()
            self.client_manager.session.auth.auth_ref = self.client_manager.auth_ref
        return

    def clean_up(self, cmd, result, err):
        self.log.debug('clean_up %s: %s', cmd.__class__.__name__, err or '')
        if hasattr(self.client_manager, 'sdk_connection'):
            self.client_manager.sdk_connection.close()
        if hasattr(self.client_manager.session, 'session'):
            self.client_manager.session.session.close()
        if self.options.timing:
            self.timing_data.extend(self.client_manager.session.get_timings())
            tcmd = timing.Timing(self, self.options)
            tparser = tcmd.get_parser('Timing')
            format = 'table'
            if hasattr(cmd, 'formatter') and cmd.formatter != cmd._formatter_plugins['table'].obj:
                format = 'csv'
            sys.stdout.write('\n')
            targs = tparser.parse_args(['-f', format])
            tcmd.run(targs)
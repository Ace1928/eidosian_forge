import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
class OpenStackMagnumShell(object):

    def get_base_parser(self):
        parser = MagnumClientArgumentParser(prog='magnum', description=__doc__.strip(), epilog='See "magnum help COMMAND" for help on a specific command.', add_help=False, formatter_class=OpenStackHelpFormatter)
        parser.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--version', action='version', version=version.version_info.version_string())
        parser.add_argument('--debug', default=False, action='store_true', help=_('Print debugging output.'))
        parser.add_argument('--os-cache', default=strutils.bool_from_string(cliutils.env('OS_CACHE', default=False)), action='store_true', help=_('Use the auth token cache. Defaults to False if env[OS_CACHE] is not set.'))
        parser.add_argument('--os-region-name', metavar='<region-name>', default=os.environ.get('OS_REGION_NAME'), help=_('Region name. Default=env[OS_REGION_NAME].'))
        parser.add_argument('--os-auth-url', metavar='<auth-auth-url>', default=cliutils.env('OS_AUTH_URL', default=None), help=_('Defaults to env[OS_AUTH_URL].'))
        parser.add_argument('--os-user-id', metavar='<auth-user-id>', default=cliutils.env('OS_USER_ID', default=None), help=_('Defaults to env[OS_USER_ID].'))
        parser.add_argument('--os-username', metavar='<auth-username>', default=cliutils.env('OS_USERNAME', default=None), help=_('Defaults to env[OS_USERNAME].'))
        parser.add_argument('--os-user-domain-id', metavar='<auth-user-domain-id>', default=cliutils.env('OS_USER_DOMAIN_ID', default=None), help=_('Defaults to env[OS_USER_DOMAIN_ID].'))
        parser.add_argument('--os-user-domain-name', metavar='<auth-user-domain-name>', default=cliutils.env('OS_USER_DOMAIN_NAME', default=None), help=_('Defaults to env[OS_USER_DOMAIN_NAME].'))
        parser.add_argument('--os-project-id', metavar='<auth-project-id>', default=cliutils.env('OS_PROJECT_ID', default=None), help=_('Defaults to env[OS_PROJECT_ID].'))
        parser.add_argument('--os-project-name', metavar='<auth-project-name>', default=cliutils.env('OS_PROJECT_NAME', default=None), help=_('Defaults to env[OS_PROJECT_NAME].'))
        parser.add_argument('--os-tenant-id', metavar='<auth-tenant-id>', default=cliutils.env('OS_TENANT_ID', default=None), help=argparse.SUPPRESS)
        parser.add_argument('--os-tenant-name', metavar='<auth-tenant-name>', default=cliutils.env('OS_TENANT_NAME', default=None), help=argparse.SUPPRESS)
        parser.add_argument('--os-project-domain-id', metavar='<auth-project-domain-id>', default=cliutils.env('OS_PROJECT_DOMAIN_ID', default=None), help=_('Defaults to env[OS_PROJECT_DOMAIN_ID].'))
        parser.add_argument('--os-project-domain-name', metavar='<auth-project-domain-name>', default=cliutils.env('OS_PROJECT_DOMAIN_NAME', default=None), help=_('Defaults to env[OS_PROJECT_DOMAIN_NAME].'))
        parser.add_argument('--os-token', metavar='<auth-token>', default=cliutils.env('OS_TOKEN', default=None), help=_('Defaults to env[OS_TOKEN].'))
        parser.add_argument('--os-password', metavar='<auth-password>', default=cliutils.env('OS_PASSWORD', default=None), help=_('Defaults to env[OS_PASSWORD].'))
        parser.add_argument('--service-type', metavar='<service-type>', help=_('Defaults to container-infra for all actions.'))
        parser.add_argument('--service_type', help=argparse.SUPPRESS)
        parser.add_argument('--endpoint-type', metavar='<endpoint-type>', default=cliutils.env('OS_ENDPOINT_TYPE', default=None), help=argparse.SUPPRESS)
        parser.add_argument('--os-endpoint-type', metavar='<os-endpoint-type>', default=cliutils.env('OS_ENDPOINT_TYPE', default=None), help=_('Defaults to env[OS_ENDPOINT_TYPE]'))
        parser.add_argument('--os-interface', metavar='<os-interface>', default=cliutils.env('OS_INTERFACE', default=DEFAULT_INTERFACE), help=argparse.SUPPRESS)
        parser.add_argument('--os-cloud', metavar='<auth-cloud>', default=cliutils.env('OS_CLOUD', default=None), help=_('Defaults to env[OS_CLOUD].'))
        parser.add_argument('--magnum-api-version', metavar='<magnum-api-ver>', default=cliutils.env('MAGNUM_API_VERSION', default='latest'), help=_('Accepts "api", defaults to env[MAGNUM_API_VERSION].'))
        parser.add_argument('--magnum_api_version', help=argparse.SUPPRESS)
        parser.add_argument('--os-cacert', metavar='<ca-certificate>', default=cliutils.env('OS_CACERT', default=None), help=_('Specify a CA bundle file to use in verifying a TLS (https) server certificate. Defaults to env[OS_CACERT].'))
        parser.add_argument('--os-endpoint-override', metavar='<endpoint-override>', default=cliutils.env('OS_ENDPOINT_OVERRIDE', default=None), help=_('Use this API endpoint instead of the Service Catalog.'))
        parser.add_argument('--bypass-url', metavar='<bypass-url>', default=cliutils.env('BYPASS_URL', default=None), dest='bypass_url', help=argparse.SUPPRESS)
        parser.add_argument('--bypass_url', help=argparse.SUPPRESS)
        parser.add_argument('--insecure', default=cliutils.env('MAGNUMCLIENT_INSECURE', default=False), action='store_true', help=_('Do not verify https connections'))
        if profiler:
            parser.add_argument('--profile', metavar='HMAC_KEY', default=cliutils.env('OS_PROFILE', default=None), help='HMAC key to use for encrypting context data for performance profiling of operation. This key should be the value of the HMAC key configured for the OSprofiler middleware in magnum; it is specified in the Magnum configuration file at "/etc/magnum/magnum.conf". Without the key, profiling will not be triggered even if OSprofiler is enabled on the server side.')
        return parser

    def get_subcommand_parser(self, version):
        parser = self.get_base_parser()
        self.subcommands = {}
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        try:
            actions_modules = {'1': shell_v1.COMMAND_MODULES}[version]
        except KeyError:
            actions_modules = shell_v1.COMMAND_MODULES
        for actions_module in actions_modules:
            self._find_actions(subparsers, actions_module)
        self._find_actions(subparsers, self)
        self._add_bash_completion_subparser(subparsers)
        return parser

    def _add_bash_completion_subparser(self, subparsers):
        subparser = subparsers.add_parser('bash_completion', add_help=False, formatter_class=OpenStackHelpFormatter)
        self.subcommands['bash_completion'] = subparser
        subparser.set_defaults(func=self.do_bash_completion)

    def _find_actions(self, subparsers, actions_module):
        for attr in (a for a in dir(actions_module) if a.startswith('do_')):
            command = attr[3:].replace('_', '-')
            callback = getattr(actions_module, attr)
            desc = callback.__doc__ or ''
            action_help = desc.strip()
            arguments = getattr(callback, 'arguments', [])
            group_args = getattr(callback, 'deprecated_groups', [])
            subparser = subparsers.add_parser(command, help=action_help, description=desc, add_help=False, formatter_class=OpenStackHelpFormatter)
            subparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
            self.subcommands[command] = subparser
            for old_info, new_info, req in group_args:
                group = subparser.add_mutually_exclusive_group(required=req)
                group.add_argument(*old_info[0], **old_info[1])
                group.add_argument(*new_info[0], **new_info[1])
            for args, kwargs in arguments:
                subparser.add_argument(*args, **kwargs)
            subparser.set_defaults(func=callback)

    def setup_debugging(self, debug):
        if debug:
            streamformat = '%(levelname)s (%(module)s:%(lineno)d) %(message)s'
            logging.basicConfig(level=logging.DEBUG, format=streamformat)
        else:
            streamformat = '%(levelname)s %(message)s'
            logging.basicConfig(level=logging.CRITICAL, format=streamformat)

    def _check_version(self, api_version):
        if api_version == 'latest':
            return LATEST_API_VERSION
        else:
            try:
                versions = tuple((int(i) for i in api_version.split('.')))
            except ValueError:
                versions = ()
            if len(versions) == 1:
                magnum_api_version = None
            elif len(versions) == 2:
                magnum_api_version = api_version
                if versions[1] == 0:
                    magnum_api_version = None
            else:
                msg = _("The requested API version %(ver)s is an unexpected format. Acceptable formats are 'X', 'X.Y', or the literal string '%(latest)s'.") % {'ver': api_version, 'latest': 'latest'}
                raise exc.CommandError(msg)
            api_major_version = versions[0]
            return (api_major_version, magnum_api_version)

    def _ensure_auth_info(self, args):
        if not cliutils.isunauthenticated(args.func):
            if not (args.os_token and (args.os_auth_url or args.os_endpoint_override)) and (not args.os_cloud):
                if not (args.os_username or args.os_user_id):
                    raise exc.CommandError('You must provide a username via either --os-username or via env[OS_USERNAME]')
                if not args.os_password:
                    raise exc.CommandError('You must provide a password via either --os-password, env[OS_PASSWORD], or prompted response')
                if not args.os_project_name and (not args.os_project_id):
                    raise exc.CommandError('You must provide a project name or project id via --os-project-name, --os-project-id, env[OS_PROJECT_NAME] or env[OS_PROJECT_ID]')
                if not args.os_auth_url:
                    raise exc.CommandError('You must provide an auth url via either --os-auth-url or via env[OS_AUTH_URL]')

    def main(self, argv):
        argv = list(argv)
        parser = self.get_base_parser()
        options, args = parser.parse_known_args(argv)
        self.setup_debugging(options.debug)
        if '--endpoint_type' in argv:
            spot = argv.index('--endpoint_type')
            argv[spot] = '--endpoint-type'
        api_major_version, magnum_api_version = self._check_version(options.magnum_api_version)
        subcommand_parser = self.get_subcommand_parser(api_major_version)
        self.parser = subcommand_parser
        if options.help or not argv:
            subcommand_parser.print_help()
            return 0
        args = subcommand_parser.parse_args(argv)
        if not hasattr(args, 'func') or args.func == self.do_help:
            self.do_help(args)
            return 0
        elif args.func == self.do_bash_completion:
            self.do_bash_completion(args)
            return 0
        if not args.service_type:
            args.service_type = DEFAULT_SERVICE_TYPE
        if args.bypass_url:
            args.os_endpoint_override = args.bypass_url
        args.os_project_id = args.os_project_id or args.os_tenant_id
        args.os_project_name = args.os_project_name or args.os_tenant_name
        self._ensure_auth_info(args)
        try:
            client = {'1': client_v1}[api_major_version]
        except KeyError:
            client = client_v1
        args.os_endpoint_type = args.os_endpoint_type or args.endpoint_type
        if args.os_endpoint_type:
            args.os_interface = args.os_endpoint_type
        if args.os_interface.endswith('URL'):
            args.os_interface = args.os_interface[:-3]
        kwargs = {}
        if profiler:
            kwargs['profile'] = args.profile
        self.cs = client.Client(cloud=args.os_cloud, user_id=args.os_user_id, username=args.os_username, password=args.os_password, auth_token=args.os_token, project_id=args.os_project_id, project_name=args.os_project_name, user_domain_id=args.os_user_domain_id, user_domain_name=args.os_user_domain_name, project_domain_id=args.os_project_domain_id, project_domain_name=args.os_project_domain_name, auth_url=args.os_auth_url, service_type=args.service_type, region_name=args.os_region_name, magnum_url=args.os_endpoint_override, interface=args.os_interface, insecure=args.insecure, api_version=args.magnum_api_version, **kwargs)
        self._check_deprecation(args.func, argv)
        try:
            args.func(self.cs, args)
        except (cliutils.DuplicateArgs, cliutils.MissingArgs):
            self.do_help(args)
            raise
        if profiler and args.profile:
            trace_id = profiler.get().get_base_id()
            print('To display trace use the command:\n\n  osprofiler trace show --html %s ' % trace_id)

    def _check_deprecation(self, func, argv):
        if not hasattr(func, 'deprecated_groups'):
            return
        for old_info, new_info, required in func.deprecated_groups:
            old_param = old_info[0][0]
            new_param = new_info[0][0]
            old_value, new_value = (None, None)
            for i in range(len(argv)):
                cur_arg = argv[i]
                if cur_arg == old_param:
                    old_value = argv[i + 1]
                elif cur_arg == new_param[0]:
                    new_value = argv[i + 1]
            if old_value and (not new_value):
                print('WARNING: The %s parameter is deprecated and will be removed in a future release. Use the %s parameter to avoid seeing this message.' % (old_param, new_param))

    def _dump_timings(self, timings):

        class Tyme(object):

            def __init__(self, url, seconds):
                self.url = url
                self.seconds = seconds
        results = [Tyme(url, end - start) for url, start, end in timings]
        total = 0.0
        for tyme in results:
            total += tyme.seconds
        results.append(Tyme('Total', total))
        cliutils.print_list(results, ['url', 'seconds'], sortby_index=None)

    def do_bash_completion(self, _args):
        """Prints arguments for bash-completion.

        Prints all of the commands and options to stdout so that the
        magnum.bash_completion script doesn't have to hard code them.
        """
        commands = set()
        options = set()
        for sc_str, sc in self.subcommands.items():
            commands.add(sc_str)
            for option in sc._optionals._option_string_actions.keys():
                options.add(option)
        commands.remove('bash-completion')
        commands.remove('bash_completion')
        print(' '.join(commands | options))

    @cliutils.arg('command', metavar='<subcommand>', nargs='?', help=_('Display help for <subcommand>.'))
    def do_help(self, args):
        """Display help about this program or one of its subcommands."""
        command = getattr(args, 'command', '')
        if command:
            if args.command in self.subcommands:
                self.subcommands[args.command].print_help()
            else:
                raise exc.CommandError("'%s' is not a valid subcommand" % args.command)
        else:
            self.parser.print_help()
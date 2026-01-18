import argparse
import logging
import os
import sys
from keystoneauth1 import loading
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client
from novaclient import exceptions as exc
import novaclient.extension
from novaclient.i18n import _
from novaclient import utils
class OpenStackComputeShell(object):
    times = []

    def __init__(self):
        self.client_logger = None

    def _append_global_identity_args(self, parser, argv):
        loading.register_session_argparse_arguments(parser)
        default_auth_plugin = 'password'
        if '--os-token' in argv:
            default_auth_plugin = 'token'
        loading.register_auth_argparse_arguments(parser, argv, default=default_auth_plugin)
        parser.set_defaults(insecure=strutils.bool_from_string(utils.env('NOVACLIENT_INSECURE', default=False)))
        parser.set_defaults(os_auth_url=utils.env('OS_AUTH_URL', 'NOVA_URL'))
        parser.set_defaults(os_username=utils.env('OS_USERNAME', 'NOVA_USERNAME'))
        parser.set_defaults(os_password=utils.env('OS_PASSWORD', 'NOVA_PASSWORD'))
        parser.set_defaults(os_project_name=utils.env('OS_PROJECT_NAME', 'OS_TENANT_NAME', 'NOVA_PROJECT_ID'))
        parser.set_defaults(os_project_id=utils.env('OS_PROJECT_ID', 'OS_TENANT_ID'))
        parser.set_defaults(os_project_domain_id=utils.env('OS_PROJECT_DOMAIN_ID'))
        parser.set_defaults(os_project_domain_name=utils.env('OS_PROJECT_DOMAIN_NAME'))
        parser.set_defaults(os_user_domain_id=utils.env('OS_USER_DOMAIN_ID'))
        parser.set_defaults(os_user_domain_name=utils.env('OS_USER_DOMAIN_NAME'))

    def get_base_parser(self, argv):
        parser = NovaClientArgumentParser(prog='nova', description=__doc__.strip(), epilog='See "nova help COMMAND" for help on a specific command.', add_help=False, formatter_class=OpenStackHelpFormatter)
        parser.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--version', action='version', version=novaclient.__version__)
        parser.add_argument('--debug', default=False, action='store_true', help=_('Print debugging output.'))
        parser.add_argument('--os-cache', default=strutils.bool_from_string(utils.env('OS_CACHE', default=False), True), action='store_true', help=_('Use the auth token cache. Defaults to False if env[OS_CACHE] is not set.'))
        parser.add_argument('--timings', default=False, action='store_true', help=_('Print call timing info.'))
        parser.add_argument('--os-region-name', metavar='<region-name>', default=utils.env('OS_REGION_NAME', 'NOVA_REGION_NAME'), help=_('Defaults to env[OS_REGION_NAME].'))
        parser.add_argument('--service-type', metavar='<service-type>', help=_('Defaults to compute for most actions.'))
        parser.add_argument('--service-name', metavar='<service-name>', default=utils.env('NOVA_SERVICE_NAME'), help=_('Defaults to env[NOVA_SERVICE_NAME].'))
        parser.add_argument('--os-endpoint-type', metavar='<endpoint-type>', dest='endpoint_type', default=utils.env('NOVA_ENDPOINT_TYPE', default=utils.env('OS_ENDPOINT_TYPE', default=DEFAULT_NOVA_ENDPOINT_TYPE)), help=_('Defaults to env[NOVA_ENDPOINT_TYPE], env[OS_ENDPOINT_TYPE] or ') + DEFAULT_NOVA_ENDPOINT_TYPE + '.')
        parser.add_argument('--os-compute-api-version', metavar='<compute-api-ver>', default=utils.env('OS_COMPUTE_API_VERSION', default=DEFAULT_OS_COMPUTE_API_VERSION), help=_('Accepts X, X.Y (where X is major and Y is minor part) or "X.latest", defaults to env[OS_COMPUTE_API_VERSION].'))
        parser.add_argument('--os-endpoint-override', metavar='<bypass-url>', dest='endpoint_override', default=utils.env('OS_ENDPOINT_OVERRIDE', 'NOVACLIENT_ENDPOINT_OVERRIDE', 'NOVACLIENT_BYPASS_URL'), help=_('Use this API endpoint instead of the Service Catalog. Defaults to env[OS_ENDPOINT_OVERRIDE].'))
        parser.set_defaults(func=self.do_help)
        parser.set_defaults(command='')
        if osprofiler_profiler:
            parser.add_argument('--profile', metavar='HMAC_KEY', default=utils.env('OS_PROFILE'), help='HMAC key to use for encrypting context data for performance profiling of operation. This key should be the value of the HMAC key configured for the OSprofiler middleware in nova; it is specified in the Nova configuration file at "/etc/nova/nova.conf". Without the key, profiling will not be triggered even if OSprofiler is enabled on the server side.')
        self._append_global_identity_args(parser, argv)
        return parser

    def get_subcommand_parser(self, version, do_help=False, argv=None):
        parser = self.get_base_parser(argv)
        self.subcommands = {}
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        actions_module = importutils.import_module('novaclient.v%s.shell' % version.ver_major)
        self._find_actions(subparsers, actions_module, version, do_help)
        self._find_actions(subparsers, self, version, do_help)
        for extension in self.extensions:
            self._find_actions(subparsers, extension.module, version, do_help)
        self._add_bash_completion_subparser(subparsers)
        return parser

    def _add_bash_completion_subparser(self, subparsers):
        subparser = subparsers.add_parser('bash_completion', add_help=False, formatter_class=OpenStackHelpFormatter)
        self.subcommands['bash_completion'] = subparser
        subparser.set_defaults(func=self.do_bash_completion)

    def _find_actions(self, subparsers, actions_module, version, do_help):
        msg = _(" (Supported by API versions '%(start)s' - '%(end)s')")
        for attr in (a for a in dir(actions_module) if a.startswith('do_')):
            command = attr[3:].replace('_', '-')
            callback = getattr(actions_module, attr)
            desc = callback.__doc__ or ''
            if hasattr(callback, 'versioned'):
                additional_msg = ''
                subs = api_versions.get_substitutions(callback)
                if do_help:
                    additional_msg = msg % {'start': subs[0].start_version.get_string(), 'end': subs[-1].end_version.get_string()}
                    if version.is_latest():
                        additional_msg += HINT_HELP_MSG
                subs = [versioned_method for versioned_method in subs if version.matches(versioned_method.start_version, versioned_method.end_version)]
                if subs:
                    callback = subs[-1].func
                else:
                    continue
                desc = callback.__doc__ or desc
                desc += additional_msg
            action_help = desc.strip()
            arguments = getattr(callback, 'arguments', [])
            groups = {}
            subparser = subparsers.add_parser(command, help=action_help, description=desc, add_help=False, formatter_class=OpenStackHelpFormatter)
            subparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
            self.subcommands[command] = subparser
            for args, kwargs in arguments:
                kwargs = kwargs.copy()
                start_version = kwargs.pop('start_version', None)
                end_version = kwargs.pop('end_version', None)
                group = kwargs.pop('group', None)
                if start_version:
                    start_version = api_versions.APIVersion(start_version)
                    if end_version:
                        end_version = api_versions.APIVersion(end_version)
                    else:
                        end_version = api_versions.APIVersion('%s.latest' % start_version.ver_major)
                    if do_help:
                        kwargs['help'] = kwargs.get('help', '') + msg % {'start': start_version.get_string(), 'end': end_version.get_string()}
                    if not version.matches(start_version, end_version):
                        continue
                if group:
                    if group not in groups:
                        groups[group] = subparser.add_mutually_exclusive_group()
                    kwargs['dest'] = kwargs.get('dest', group)
                    groups[group].add_argument(*args, **kwargs)
                else:
                    subparser.add_argument(*args, **kwargs)
            subparser.set_defaults(func=callback)

    def setup_debugging(self, debug):
        if not debug:
            return
        streamformat = '%(levelname)s (%(module)s:%(lineno)d) %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=streamformat)
        logging.getLogger('iso8601').setLevel(logging.WARNING)
        self.client_logger = logging.getLogger(client.__name__)
        ch = logging.StreamHandler()
        self.client_logger.setLevel(logging.DEBUG)
        self.client_logger.addHandler(ch)

    def main(self, argv):
        parser = self.get_base_parser(argv)
        args, args_list = parser.parse_known_args(argv)
        self.setup_debugging(args.debug)
        self.extensions = []
        do_help = args.help or not args_list or args_list[0] == 'help'
        skip_auth = do_help or 'bash-completion' in argv
        if not args.os_compute_api_version:
            api_version = api_versions.get_api_version(DEFAULT_MAJOR_OS_COMPUTE_API_VERSION)
        else:
            api_version = api_versions.get_api_version(args.os_compute_api_version)
        auth_token = getattr(args, 'os_token', None)
        os_username = getattr(args, 'os_username', None)
        os_user_id = getattr(args, 'os_user_id', None)
        os_password = None
        os_project_name = getattr(args, 'os_project_name', getattr(args, 'os_tenant_name', None))
        os_project_id = getattr(args, 'os_project_id', getattr(args, 'os_tenant_id', None))
        os_auth_url = args.os_auth_url
        os_region_name = args.os_region_name
        if 'v2.0' not in os_auth_url:
            if not args.os_project_domain_id and (not args.os_project_domain_name):
                setattr(args, 'os_project_domain_id', 'default')
            if not auth_token and (not args.os_user_domain_id and (not args.os_user_domain_name)):
                setattr(args, 'os_user_domain_id', 'default')
        os_project_domain_id = args.os_project_domain_id
        os_project_domain_name = args.os_project_domain_name
        os_user_domain_id = getattr(args, 'os_user_domain_id', None)
        os_user_domain_name = getattr(args, 'os_user_domain_name', None)
        endpoint_type = args.endpoint_type
        insecure = args.insecure
        service_type = args.service_type
        service_name = args.service_name
        endpoint_override = args.endpoint_override
        os_cache = args.os_cache
        cacert = args.os_cacert
        cert = args.os_cert
        timeout = args.timeout
        keystone_session = None
        keystone_auth = None
        if not endpoint_type:
            endpoint_type = DEFAULT_NOVA_ENDPOINT_TYPE
        if endpoint_type in ['internal', 'public', 'admin']:
            endpoint_type += 'URL'
        if not service_type:
            service_type = DEFAULT_NOVA_SERVICE_TYPE
        must_auth = not (auth_token and endpoint_override)
        if must_auth and (not skip_auth):
            if not any([auth_token, os_username, os_user_id]):
                raise exc.CommandError(_('You must provide a user name/id (via --os-username, --os-user-id, env[OS_USERNAME] or env[OS_USER_ID]) or an auth token (via --os-token).'))
            if not any([os_project_name, os_project_id]):
                raise exc.CommandError(_('You must provide a project name or project ID via --os-project-name, --os-project-id, env[OS_PROJECT_ID] or env[OS_PROJECT_NAME]. You may use os-project and os-tenant interchangeably.'))
            if not os_auth_url:
                raise exc.CommandError(_('You must provide an auth url via either --os-auth-url or env[OS_AUTH_URL].'))
            with utils.record_time(self.times, args.timings, 'auth_url', args.os_auth_url):
                keystone_session = loading.load_session_from_argparse_arguments(args)
                keystone_auth = loading.load_auth_from_argparse_arguments(args)
        if not skip_auth and (not any([os_project_name, os_project_id])):
            raise exc.CommandError(_('You must provide a project name or project id via --os-project-name, --os-project-id, env[OS_PROJECT_ID] or env[OS_PROJECT_NAME]. You may use os-project and os-tenant interchangeably.'))
        if not os_auth_url and (not skip_auth):
            raise exc.CommandError(_('You must provide an auth url via either --os-auth-url or env[OS_AUTH_URL]'))
        additional_kwargs = {}
        if osprofiler_profiler:
            additional_kwargs['profile'] = args.profile
        self.cs = client.Client(api_versions.APIVersion('2.0'), os_username, os_password, project_id=os_project_id, project_name=os_project_name, user_id=os_user_id, auth_url=os_auth_url, insecure=insecure, region_name=os_region_name, endpoint_type=endpoint_type, extensions=self.extensions, service_type=service_type, service_name=service_name, auth_token=auth_token, timings=args.timings, endpoint_override=endpoint_override, os_cache=os_cache, http_log_debug=args.debug, cacert=cacert, cert=cert, timeout=timeout, session=keystone_session, auth=keystone_auth, logger=self.client_logger, project_domain_id=os_project_domain_id, project_domain_name=os_project_domain_name, user_domain_id=os_user_domain_id, user_domain_name=os_user_domain_name, **additional_kwargs)
        if not skip_auth:
            if not api_version.is_latest():
                if api_version > api_versions.APIVersion('2.0'):
                    if not api_version.matches(novaclient.API_MIN_VERSION, novaclient.API_MAX_VERSION):
                        raise exc.CommandError(_("The specified version isn't supported by client. The valid version range is '%(min)s' to '%(max)s'") % {'min': novaclient.API_MIN_VERSION.get_string(), 'max': novaclient.API_MAX_VERSION.get_string()})
            api_version = api_versions.discover_version(self.cs, api_version)
        self.extensions = client.discover_extensions(api_version)
        self._run_extension_hooks('__pre_parse_args__')
        subcommand_parser = self.get_subcommand_parser(api_version, do_help=do_help, argv=argv)
        self.parser = subcommand_parser
        if args.help or not argv:
            subcommand_parser.print_help()
            return 0
        args = subcommand_parser.parse_args(argv)
        self._run_extension_hooks('__post_parse_args__', args)
        if args.func == self.do_help:
            self.do_help(args)
            return 0
        elif args.func == self.do_bash_completion:
            self.do_bash_completion(args)
            return 0
        if not args.service_type:
            service_type = utils.get_service_type(args.func) or DEFAULT_NOVA_SERVICE_TYPE
        if utils.isunauthenticated(args.func):
            keystone_session = None
            keystone_auth = None
        self.cs = client.Client(api_version, os_username, os_password, project_id=os_project_id, project_name=os_project_name, user_id=os_user_id, auth_url=os_auth_url, insecure=insecure, region_name=os_region_name, endpoint_type=endpoint_type, extensions=self.extensions, service_type=service_type, service_name=service_name, auth_token=auth_token, timings=args.timings, endpoint_override=endpoint_override, os_cache=os_cache, http_log_debug=args.debug, cacert=cacert, cert=cert, timeout=timeout, session=keystone_session, auth=keystone_auth, project_domain_id=os_project_domain_id, project_domain_name=os_project_domain_name, user_domain_id=os_user_domain_id, user_domain_name=os_user_domain_name)
        args.func(self.cs, args)
        if osprofiler_profiler and args.profile:
            trace_id = osprofiler_profiler.get().get_base_id()
            print('To display trace use the command:\n\n  osprofiler trace show --html %s ' % trace_id)
        if args.timings:
            self._dump_timings(self.times + self.cs.get_timings())

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
        utils.print_list(results, ['url', 'seconds'], sortby_index=None)

    def _run_extension_hooks(self, hook_type, *args, **kwargs):
        """Run hooks for all registered extensions."""
        for extension in self.extensions:
            extension.run_hooks(hook_type, *args, **kwargs)

    def do_bash_completion(self, _args):
        """
        Prints all of the commands and options to stdout so that the
        nova.bash_completion script doesn't have to hard code them.
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

    @utils.arg('command', metavar='<subcommand>', nargs='?', help=_('Display help for <subcommand>.'))
    def do_help(self, args):
        """
        Display help about this program or one of its subcommands.
        """
        if args.command:
            if args.command in self.subcommands:
                self.subcommands[args.command].print_help()
            else:
                raise exc.CommandError(_("'%s' is not a valid subcommand") % args.command)
        else:
            self.parser.print_help()
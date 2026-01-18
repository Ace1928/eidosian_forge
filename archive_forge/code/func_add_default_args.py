import argparse
import getpass
import io
import json
import logging
import signal
import socket
import warnings
from os import environ, walk, _exit as os_exit
from os.path import isfile, isdir, join
from urllib.parse import unquote, urlparse
from sys import argv as sys_argv, exit, stderr, stdin
from time import gmtime, strftime
from swiftclient import RequestException
from swiftclient.utils import config_true_value, generate_temp_url, \
from swiftclient.multithreading import OutputManager
from swiftclient.exceptions import ClientException
from swiftclient import __version__ as client_version
from swiftclient.client import logger_settings as client_logger_settings, \
from swiftclient.service import SwiftService, SwiftError, \
from swiftclient.command_helpers import print_account_stats, \
def add_default_args(parser):
    default_auth_version = '1.0'
    for k in ('ST_AUTH_VERSION', 'OS_AUTH_VERSION', 'OS_IDENTITY_API_VERSION'):
        try:
            default_auth_version = environ[k]
            break
        except KeyError:
            pass
    parser.add_argument('--os-help', action='store_true', dest='os_help', help='Show OpenStack authentication options.')
    parser.add_argument('--os_help', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-s', '--snet', action='store_true', dest='snet', default=False, help='Use SERVICENET internal network.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=1, help='Print more info.')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False, help='Show the curl commands and results of all http queries regardless of result status.')
    parser.add_argument('--info', action='store_true', dest='info', default=False, help='Show the curl commands and results of all http queries which return an error.')
    parser.add_argument('-q', '--quiet', action='store_const', dest='verbose', const=0, default=1, help='Suppress status output.')
    parser.add_argument('-A', '--auth', dest='auth', default=environ.get('ST_AUTH'), help='URL for obtaining an auth token.')
    parser.add_argument('-V', '--auth-version', '--os-identity-api-version', dest='auth_version', default=default_auth_version, type=str, help='Specify a version for authentication. Defaults to env[ST_AUTH_VERSION], env[OS_AUTH_VERSION], env[OS_IDENTITY_API_VERSION] or 1.0.')
    parser.add_argument('-U', '--user', dest='user', default=environ.get('ST_USER'), help='User name for obtaining an auth token.')
    parser.add_argument('-K', '--key', dest='key', default=environ.get('ST_KEY'), help='Key for obtaining an auth token.')
    parser.add_argument('-T', '--timeout', type=parse_timeout, dest='timeout', default=None, help='Timeout in seconds to wait for response.')
    parser.add_argument('-R', '--retries', type=int, default=5, dest='retries', help='The number of times to retry a failed connection.')
    default_val = config_true_value(environ.get('SWIFTCLIENT_INSECURE'))
    parser.add_argument('--insecure', action='store_true', dest='insecure', default=default_val, help="Allow swiftclient to access servers without having to verify the SSL certificate. Defaults to env[SWIFTCLIENT_INSECURE] (set to 'true' to enable).")
    parser.add_argument('--no-ssl-compression', action='store_false', dest='ssl_compression', default=True, help='This option is deprecated and not used anymore. SSL compression should be disabled by default by the system SSL library.')
    parser.add_argument('--force-auth-retry', action='store_true', dest='force_auth_retry', default=False, help='Force a re-auth attempt on any error other than 401 unauthorized')
    parser.add_argument('--prompt', action='store_true', dest='prompt', default=False, help='Prompt user to enter a password which overrides any password supplied via --key, --os-password or environment variables.')
    os_grp = parser.add_argument_group('OpenStack authentication options')
    os_grp.add_argument('--os-username', metavar='<auth-user-name>', default=environ.get('OS_USERNAME'), help='OpenStack username. Defaults to env[OS_USERNAME].')
    os_grp.add_argument('--os_username', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-user-id', metavar='<auth-user-id>', default=environ.get('OS_USER_ID'), help='OpenStack user ID. Defaults to env[OS_USER_ID].')
    os_grp.add_argument('--os_user_id', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-user-domain-id', metavar='<auth-user-domain-id>', default=environ.get('OS_USER_DOMAIN_ID'), help='OpenStack user domain ID. Defaults to env[OS_USER_DOMAIN_ID].')
    os_grp.add_argument('--os_user_domain_id', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-user-domain-name', metavar='<auth-user-domain-name>', default=environ.get('OS_USER_DOMAIN_NAME'), help='OpenStack user domain name. Defaults to env[OS_USER_DOMAIN_NAME].')
    os_grp.add_argument('--os_user_domain_name', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-password', metavar='<auth-password>', default=environ.get('OS_PASSWORD'), help='OpenStack password. Defaults to env[OS_PASSWORD].')
    os_grp.add_argument('--os_password', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-tenant-id', metavar='<auth-tenant-id>', default=environ.get('OS_TENANT_ID'), help='OpenStack tenant ID. Defaults to env[OS_TENANT_ID].')
    os_grp.add_argument('--os_tenant_id', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-tenant-name', metavar='<auth-tenant-name>', default=environ.get('OS_TENANT_NAME'), help='OpenStack tenant name. Defaults to env[OS_TENANT_NAME].')
    os_grp.add_argument('--os_tenant_name', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-project-id', metavar='<auth-project-id>', default=environ.get('OS_PROJECT_ID'), help='OpenStack project ID. Defaults to env[OS_PROJECT_ID].')
    os_grp.add_argument('--os_project_id', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-project-name', metavar='<auth-project-name>', default=environ.get('OS_PROJECT_NAME'), help='OpenStack project name. Defaults to env[OS_PROJECT_NAME].')
    os_grp.add_argument('--os_project_name', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-project-domain-id', metavar='<auth-project-domain-id>', default=environ.get('OS_PROJECT_DOMAIN_ID'), help='OpenStack project domain ID. Defaults to env[OS_PROJECT_DOMAIN_ID].')
    os_grp.add_argument('--os_project_domain_id', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-project-domain-name', metavar='<auth-project-domain-name>', default=environ.get('OS_PROJECT_DOMAIN_NAME'), help='OpenStack project domain name. Defaults to env[OS_PROJECT_DOMAIN_NAME].')
    os_grp.add_argument('--os_project_domain_name', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-auth-url', metavar='<auth-url>', default=environ.get('OS_AUTH_URL'), help='OpenStack auth URL. Defaults to env[OS_AUTH_URL].')
    os_grp.add_argument('--os_auth_url', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-auth-type', metavar='<auth-type>', default=environ.get('OS_AUTH_TYPE'), help='OpenStack auth type for v3. Defaults to env[OS_AUTH_TYPE].')
    os_grp.add_argument('--os_auth_type', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-application-credential-id', metavar='<auth-application-credential-id>', default=environ.get('OS_APPLICATION_CREDENTIAL_ID'), help='OpenStack appplication credential id. Defaults to env[OS_APPLICATION_CREDENTIAL_ID].')
    os_grp.add_argument('--os_application_credential_id', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-application-credential-secret', metavar='<auth-application-credential-secret>', default=environ.get('OS_APPLICATION_CREDENTIAL_SECRET'), help='OpenStack appplication credential secret. Defaults to env[OS_APPLICATION_CREDENTIAL_SECRET].')
    os_grp.add_argument('--os_application_credential_secret', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-auth-token', metavar='<auth-token>', default=environ.get('OS_AUTH_TOKEN'), help='OpenStack token. Defaults to env[OS_AUTH_TOKEN]. Used with --os-storage-url to bypass the usual username/password authentication.')
    os_grp.add_argument('--os_auth_token', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-storage-url', metavar='<storage-url>', default=environ.get('OS_STORAGE_URL'), help='OpenStack storage URL. Defaults to env[OS_STORAGE_URL]. Overrides the storage url returned during auth. Will bypass authentication when used with --os-auth-token.')
    os_grp.add_argument('--os_storage_url', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-region-name', metavar='<region-name>', default=environ.get('OS_REGION_NAME'), help='OpenStack region name. Defaults to env[OS_REGION_NAME].')
    os_grp.add_argument('--os_region_name', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-service-type', metavar='<service-type>', default=environ.get('OS_SERVICE_TYPE'), help='OpenStack Service type. Defaults to env[OS_SERVICE_TYPE].')
    os_grp.add_argument('--os_service_type', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-endpoint-type', metavar='<endpoint-type>', default=environ.get('OS_ENDPOINT_TYPE'), help='OpenStack Endpoint type. Defaults to env[OS_ENDPOINT_TYPE].')
    os_grp.add_argument('--os_endpoint_type', help=argparse.SUPPRESS)
    os_grp.add_argument('--os-cacert', metavar='<ca-certificate>', default=environ.get('OS_CACERT'), help='Specify a CA bundle file to use in verifying a TLS (https) server certificate. Defaults to env[OS_CACERT].')
    os_grp.add_argument('--os-cert', metavar='<client-certificate-file>', default=environ.get('OS_CERT'), help='Specify a client certificate file (for client auth). Defaults to env[OS_CERT].')
    os_grp.add_argument('--os-key', metavar='<client-certificate-key-file>', default=environ.get('OS_KEY'), help='Specify a client certificate key file (for client auth). Defaults to env[OS_KEY].')
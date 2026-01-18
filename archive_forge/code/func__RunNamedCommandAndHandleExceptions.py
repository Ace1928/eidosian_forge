from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import errno
import getopt
import logging
import os
import re
import signal
import socket
import sys
import textwrap
import traceback
import six
from six.moves import configparser
from six.moves import range
from google.auth import exceptions as google_auth_exceptions
import gslib.exception
from gslib.exception import CommandException
from gslib.exception import ControlCException
from gslib.utils.version_check import check_python_version_support
from gslib.utils.arg_helper import GetArgumentsAndOptions
from gslib.utils.user_agent_helper import GetUserAgent
import boto
import gslib
from gslib.utils import system_util, text_util
from gslib import metrics
import httplib2
import oauth2client
from google_reauth import reauth_creds
from google_reauth import errors as reauth_errors
from gslib import context_config
from gslib import wildcard_iterator
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import ProjectIdException
from gslib.cloud_api import ServiceException
from gslib.command_runner import CommandRunner
import apitools.base.py.exceptions as apitools_exceptions
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import InitializeSignalHandling
from gslib.sig_handling import RegisterSignalHandler
def _RunNamedCommandAndHandleExceptions(command_runner, command_name, args=None, headers=None, debug_level=0, trace_token=None, parallel_operations=False, perf_trace_token=None, user_project=None):
    """Runs the command and handles common exceptions."""
    try:
        RegisterSignalHandler(signal.SIGINT, _HandleControlC, is_final_handler=True)
        if not system_util.IS_WINDOWS:
            RegisterSignalHandler(signal.SIGQUIT, _HandleSigQuit)
        return command_runner.RunNamedCommand(command_name, args, headers, debug_level, trace_token, parallel_operations, perf_trace_token=perf_trace_token, collect_analytics=True, user_project=user_project)
    except AttributeError as e:
        if str(e).find('secret_access_key') != -1:
            _OutputAndExit('Missing credentials for the given URI(s). Does your boto config file contain all needed credentials?', exception=e)
        else:
            _OutputAndExit(message=str(e), exception=e)
    except CommandException as e:
        _HandleCommandException(e)
    except getopt.GetoptError as e:
        _HandleCommandException(CommandException(e.msg))
    except boto.exception.InvalidUriError as e:
        _OutputAndExit(message='InvalidUriError: %s.' % e.message, exception=e)
    except gslib.exception.InvalidUrlError as e:
        _OutputAndExit(message='InvalidUrlError: %s.' % e.message, exception=e)
    except boto.auth_handler.NotReadyToAuthenticate as e:
        _OutputAndExit(message='NotReadyToAuthenticate', exception=e)
    except gslib.exception.ExternalBinaryError as e:
        _OutputAndExit(message=str(e), exception=e)
    except OSError as e:
        if e.errno == errno.EPIPE or ((system_util.IS_WINDOWS and e.errno == errno.EINVAL) and (not system_util.IsRunningInteractively())):
            sys.exit(0)
        else:
            _OutputAndExit(message='OSError: %s.' % e.strerror, exception=e)
    except IOError as e:
        if e.errno == errno.EPIPE or ((system_util.IS_WINDOWS and e.errno == errno.EINVAL) and (not system_util.IsRunningInteractively())):
            sys.exit(0)
        else:
            raise
    except wildcard_iterator.WildcardException as e:
        _OutputAndExit(message=e.reason, exception=e)
    except ProjectIdException as e:
        _OutputAndExit('You are attempting to perform an operation that requires a project id, with none configured. Please re-run gsutil config and make sure to follow the instructions for finding and entering your default project id.', exception=e)
    except BadRequestException as e:
        if e.reason == 'MissingSecurityHeader':
            _CheckAndHandleCredentialException(e, args)
        _OutputAndExit(message=e, exception=e)
    except AccessDeniedException as e:
        _CheckAndHandleCredentialException(e, args)
        _OutputAndExit(message=e, exception=e)
    except ArgumentException as e:
        _OutputAndExit(message=e, exception=e)
    except ServiceException as e:
        _OutputAndExit(message=e, exception=e)
    except (oauth2client.client.HttpAccessTokenRefreshError, google_auth_exceptions.OAuthError) as e:
        if system_util.InvokedViaCloudSdk():
            _OutputAndExit('Your credentials are invalid. Please run\n$ gcloud auth login', exception=e)
        else:
            _OutputAndExit('Your credentials are invalid. For more help, see "gsutil help creds", or re-run the gsutil config command (see "gsutil help config").', exception=e)
    except apitools_exceptions.HttpError as e:
        _OutputAndExit('HttpError: %s, %s' % (getattr(e.response, 'status', ''), e.content or ''), exception=e)
    except socket.error as e:
        if e.args[0] == errno.EPIPE:
            _OutputAndExit('Got a "Broken pipe" error. This can happen to clients using Python 2.x, when the server sends an error response and then closes the socket (see http://bugs.python.org/issue5542). If you are trying to upload a large object you might retry with a small (say 200k) object, and see if you get a more specific error code.', exception=e)
        elif e.args[0] == errno.ECONNRESET and ' '.join(args).contains('s3://'):
            _OutputAndExit('\n'.join(textwrap.wrap('Got a "Connection reset by peer" error. One way this can happen is when copying data to/from an S3 regional bucket. If you are using a regional S3 bucket you could try re-running this command using the regional S3 endpoint, for example s3://s3-<region>.amazonaws.com/your-bucket. For details about this problem see https://github.com/boto/boto/issues/2207')), exception=e)
        else:
            _HandleUnknownFailure(e)
    except oauth2client.client.FlowExchangeError as e:
        _OutputAndExit('\n%s\n\n' % '\n'.join(textwrap.wrap("Failed to retrieve valid credentials (%s). Make sure you selected and pasted the ENTIRE authorization code (including any numeric prefix e.g. '4/')." % e)), exception=e)
    except reauth_errors.ReauthSamlLoginRequiredError:
        if system_util.InvokedViaCloudSdk():
            _OutputAndExit('You must re-authenticate with your SAML IdP. Please run\n$ gcloud auth login')
        else:
            _OutputAndExit('You must re-authenticate with your SAML IdP. Please run\n$ gsutil config')
    except Exception as e:
        config_paths = ', '.join(boto_util.GetFriendlyConfigFilePaths())
        if 'mac verify failure' in str(e):
            _OutputAndExit('Encountered an error while refreshing access token. If you are using a service account,\nplease verify that the gs_service_key_file_password field in your config file(s),\n%s, is correct.' % config_paths, exception=e)
        elif 'asn1 encoding routines' in str(e):
            _OutputAndExit('Encountered an error while refreshing access token. If you are using a service account,\nplease verify that the gs_service_key_file field in your config file(s),\n%s, is correct.' % config_paths, exception=e)
        _HandleUnknownFailure(e)
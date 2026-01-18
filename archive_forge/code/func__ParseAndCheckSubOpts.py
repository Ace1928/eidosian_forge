from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
import copy
from datetime import datetime
from datetime import timedelta
import getpass
import json
import re
import sys
import six
from six.moves import urllib
from apitools.base.py.exceptions import HttpError
from apitools.base.py.http_wrapper import MakeRequest
from apitools.base.py.http_wrapper import Request
from boto import config
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils import constants
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.shim_util import GcloudStorageMap, GcloudStorageFlag
from gslib.utils.signurl_helper import CreatePayload, GetFinalUrl
def _ParseAndCheckSubOpts(self):
    delta = None
    method = 'GET'
    content_type = ''
    passwd = None
    region = _AUTO_DETECT_REGION
    use_service_account = False
    billing_project = None
    for o, v in self.sub_opts:
        if six.PY2:
            v = v.decode(sys.stdin.encoding or constants.UTF8)
        if o == '-d':
            if delta is not None:
                delta += _DurationToTimeDelta(v)
            else:
                delta = _DurationToTimeDelta(v)
        elif o == '-m':
            method = v
        elif o == '-c':
            content_type = v
        elif o == '-p':
            passwd = v
        elif o == '-r':
            region = v
        elif o == '-u' or o == '--use-service-account':
            use_service_account = True
        elif o == '-b':
            billing_project = v
        else:
            self.RaiseInvalidArgumentException()
    if delta is None:
        delta = timedelta(hours=1)
    elif use_service_account and delta > _MAX_EXPIRATION_TIME_WITH_MINUS_U:
        raise CommandException('Max valid duration allowed is %s when -u flag is used. For longer duration, consider using the private-key-file instead of the -u option.' % _MAX_EXPIRATION_TIME_WITH_MINUS_U)
    elif delta > _MAX_EXPIRATION_TIME:
        raise CommandException('Max valid duration allowed is %s' % _MAX_EXPIRATION_TIME)
    if method not in ['GET', 'PUT', 'DELETE', 'HEAD', 'RESUMABLE']:
        raise CommandException('HTTP method must be one of[GET|HEAD|PUT|DELETE|RESUMABLE]')
    if not use_service_account and len(self.args) < 2:
        raise CommandException('The command requires a key file argument and one or more URL arguments if the --use-service-account flag is missing. Run `gsutil help signurl` for more info')
    if use_service_account and billing_project:
        raise CommandException('Specifying both the -b and --use-service-account options together isinvalid.')
    return (method, delta, content_type, passwd, region, use_service_account, billing_project)
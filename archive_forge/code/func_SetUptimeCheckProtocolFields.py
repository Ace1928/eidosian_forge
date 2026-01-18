from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def SetUptimeCheckProtocolFields(args, messages, uptime_check, headers, status_classes, status_codes, update=False):
    """Set Protocol fields based on args."""
    if not update and args.IsSpecified('synthetic_target') or uptime_check.syntheticMonitor is not None:
        should_not_be_set = ['--path', '--validate-ssl', '--mask-headers', '--custom-content-type', '--username', '--password', '--body', '--request-method', '--content-type', '--port', '--pings-count']
        for flag in should_not_be_set:
            dest = _FlagToDest(flag)
            if args.IsSpecified(dest):
                raise calliope_exc.InvalidArgumentException(flag, 'Should not be set for Synthetic Monitor.')
        if headers:
            raise calliope_exc.InvalidArgumentException('headers', 'Should not be set or updated for Synthetic Monitor.')
        if status_classes:
            raise calliope_exc.InvalidArgumentException('status-classes', 'Should not be set or updated for Synthetic Monitor.')
        if status_codes:
            raise calliope_exc.InvalidArgumentException('status-codes', 'Should not be set or updated for Synthetic Monitor.')
        return
    if not update and args.protocol == 'tcp' or uptime_check.tcpCheck is not None:
        if args.port is None and uptime_check.tcpCheck is None:
            raise MissingRequiredFieldError('Missing required field "port"')
        if uptime_check.tcpCheck is None:
            uptime_check.tcpCheck = messages.TcpCheck()
        tcp_check = uptime_check.tcpCheck
        if args.port is not None:
            tcp_check.port = args.port
        if args.pings_count is not None:
            tcp_check.pingConfig = messages.PingConfig()
            tcp_check.pingConfig.pingsCount = args.pings_count
        should_not_be_set = ['--path', '--validate-ssl', '--mask-headers', '--custom-content-type', '--username', '--password', '--body', '--request-method', '--content-type']
        for flag in should_not_be_set:
            dest = _FlagToDest(flag)
            if args.IsSpecified(dest):
                raise calliope_exc.InvalidArgumentException(flag, 'Should not be set for TCP Uptime Check.')
        if headers:
            raise calliope_exc.InvalidArgumentException('headers', 'Should not be set or updated for TCP Uptime Check.')
        if status_classes:
            raise calliope_exc.InvalidArgumentException('status-classes', 'Should not be set or updated for TCP Uptime Check.')
        if status_codes:
            raise calliope_exc.InvalidArgumentException('status-codes', 'Should not be set or updated for TCP Uptime Check.')
    else:
        if uptime_check.httpCheck is None:
            uptime_check.httpCheck = messages.HttpCheck()
        http_check = uptime_check.httpCheck
        if args.path is not None:
            http_check.path = args.path
        if args.validate_ssl is not None:
            http_check.validateSsl = args.validate_ssl
        if args.mask_headers is not None:
            http_check.maskHeaders = args.mask_headers
        if args.custom_content_type is not None:
            http_check.customContentType = args.custom_content_type
        if http_check.authInfo is None:
            http_check.authInfo = messages.BasicAuthentication()
        if args.username is not None:
            http_check.authInfo.username = args.username
        if args.password is not None:
            http_check.authInfo.password = args.password
        if args.pings_count is not None:
            http_check.pingConfig = messages.PingConfig()
            http_check.pingConfig.pingsCount = args.pings_count
        if args.body is not None:
            http_check.body = args.body.encode()
        if not update and args.protocol == 'https' or http_check.useSsl:
            http_check.useSsl = True
            if args.port is not None:
                http_check.port = args.port
            if http_check.port is None:
                http_check.port = 443
        else:
            http_check.useSsl = False
            if args.port is not None:
                http_check.port = args.port
            if http_check.port is None:
                http_check.port = 80
        method_mapping = {'get': messages.HttpCheck.RequestMethodValueValuesEnum.GET, 'post': messages.HttpCheck.RequestMethodValueValuesEnum.POST, None: messages.HttpCheck.RequestMethodValueValuesEnum.GET}
        if http_check.requestMethod is None or args.request_method is not None:
            http_check.requestMethod = method_mapping.get(args.request_method)
        content_mapping = {'unspecified': messages.HttpCheck.ContentTypeValueValuesEnum.TYPE_UNSPECIFIED, 'url-encoded': messages.HttpCheck.ContentTypeValueValuesEnum.URL_ENCODED, 'user-provided': messages.HttpCheck.ContentTypeValueValuesEnum.USER_PROVIDED, None: messages.HttpCheck.ContentTypeValueValuesEnum.TYPE_UNSPECIFIED}
        if http_check.contentType is None or args.content_type is not None:
            http_check.contentType = content_mapping.get(args.content_type)
        http_check.headers = headers
        status_mapping = {'1xx': messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_1XX, '2xx': messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_2XX, '3xx': messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_3XX, '4xx': messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_4XX, '5xx': messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_5XX, 'any': messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_ANY, None: messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_UNSPECIFIED}
        if status_classes is not None:
            http_check.acceptedResponseStatusCodes = []
            for status in status_classes:
                http_check.acceptedResponseStatusCodes.append(messages.ResponseStatusCode(statusClass=status_mapping.get(status)))
        elif status_codes is not None:
            http_check.acceptedResponseStatusCodes = []
            for status in status_codes:
                http_check.acceptedResponseStatusCodes.append(messages.ResponseStatusCode(statusValue=status))
        elif http_check.acceptedResponseStatusCodes is None:
            http_check.acceptedResponseStatusCodes = []
            http_check.acceptedResponseStatusCodes.append(messages.ResponseStatusCode(statusClass=status_mapping.get('2xx')))
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def CreateServiceAccountMessages(messages, scopes, service_account):
    """Returns a list of ServiceAccount messages corresponding to scopes."""
    if scopes is None:
        scopes = constants.DEFAULT_SCOPES
    if service_account is None:
        service_account = 'default'
    accounts_to_scopes = collections.defaultdict(list)
    for scope in scopes:
        parts = scope.split('=')
        if len(parts) == 1:
            account = service_account
            scope_uri = scope
        elif len(parts) == 2:
            raise calliope_exceptions.InvalidArgumentException('--scopes', 'Flag format --scopes [ACCOUNT=]SCOPE,[[ACCOUNT=]SCOPE, ...] is removed. Use --scopes [SCOPE,...] --service-account ACCOUNT instead.')
        else:
            raise calliope_exceptions.InvalidArgumentException('--scopes', '[{0}] is an illegal value for [--scopes]. Values must be of the form [SCOPE].'.format(scope))
        if service_account != 'default' and (not EMAIL_REGEX.match(service_account)):
            raise calliope_exceptions.InvalidArgumentException('--service-account', 'Invalid format: expected default or user@domain.com, received ' + service_account)
        scope_uri = constants.SCOPES.get(scope_uri, [scope_uri])
        accounts_to_scopes[account].extend(scope_uri)
    if not scopes and service_account != 'default':
        return [messages.ServiceAccount(email=service_account, scopes=[])]
    res = []
    for account, scopes in sorted(six.iteritems(accounts_to_scopes)):
        res.append(messages.ServiceAccount(email=account, scopes=sorted(scopes)))
    return res
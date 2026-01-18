from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import exceptions
def ValidateURI(uri, recovery_only):
    if (uri is None or not uri) and (not recovery_only):
        raise sql_exceptions.ArgumentError('missing URI arg, please include URI arg or set the recovery-only flag if you meant to bring database online only\n')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetUpdatedConnectionProfile(self, connection_profile, args):
    """Returns updated connection profile and list of updated fields."""
    update_fields = []
    if args.IsSpecified('display_name'):
        connection_profile.displayName = args.display_name
        update_fields.append('displayName')
    if connection_profile.mysql is not None:
        self._UpdateMySqlConnectionProfile(connection_profile, args, update_fields)
    elif self._SupportsPostgresql() and connection_profile.postgresql is not None:
        self._UpdatePostgreSqlConnectionProfile(connection_profile, args, update_fields)
    elif self._SupportsOracle() and connection_profile.oracle is not None:
        self._UpdateOracleConnectionProfile(connection_profile, args, update_fields)
    else:
        raise UnsupportedConnectionProfileDBTypeError('The requested connection profile does not contain a MySQL, PostgreSQL or Oracle object. Currently only MySQL, PostgreSQL and Oracle connection profiles are supported.')
    self._UpdateLabels(connection_profile, args)
    return (connection_profile, update_fields)
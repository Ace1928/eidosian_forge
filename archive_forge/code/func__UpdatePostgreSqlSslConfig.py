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
def _UpdatePostgreSqlSslConfig(self, connection_profile, args, update_fields):
    """Fills connection_profile and update_fields with PostgreSQL SSL data from args."""
    if args.IsSpecified('ca_certificate'):
        connection_profile.postgresql.ssl.caCertificate = args.ca_certificate
        update_fields.append('postgresql.ssl.caCertificate')
    if args.IsSpecified('private_key'):
        connection_profile.postgresql.ssl.clientKey = args.private_key
        update_fields.append('postgresql.ssl.clientKey')
    if args.IsSpecified(self._ClientCertificateArgName()):
        connection_profile.postgresql.ssl.clientCertificate = args.GetValue(self._ClientCertificateArgName())
        update_fields.append('postgresql.ssl.clientCertificate')
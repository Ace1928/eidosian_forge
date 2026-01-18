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
def _UpdateOracleSslConfig(self, connection_profile, args, update_fields):
    """Fills connection_profile and update_fields with Oracle SSL data from args."""
    if args.IsSpecified('ca_certificate'):
        connection_profile.oracle.ssl.caCertificate = args.ca_certificate
        update_fields.append('postgresql.ssl.caCertificate')
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
def _GetEngineFromCloudSql(self, cloudsql):
    """Gets the SQL engine from the Cloud SQL version.

    Args:
      cloudsql: Cloud SQL connection profile

    Returns:
      A string representing the SQL engine
    """
    if cloudsql.settings.databaseVersion:
        return '{}'.format(cloudsql.settings.databaseVersion).split('_')[0]
    else:
        return ''
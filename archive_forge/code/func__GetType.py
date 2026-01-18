from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import connection_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastream import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def _GetType(self, profile):
    if profile.mysqlProfile:
        return 'MySQL'
    elif profile.oracleProfile:
        return 'Oracle'
    elif profile.postgresqlProfile:
        return 'PostgreSQL'
    elif profile.gcsProfile:
        return 'Google Cloud Storage'
    elif profile.sqlServerProfile:
        return 'SQL Server'
    elif profile.bigqueryProfile:
        return 'BigQuery'
    else:
        return None
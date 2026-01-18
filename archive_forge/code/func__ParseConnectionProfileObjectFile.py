from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _ParseConnectionProfileObjectFile(self, connection_profile_object_file, release_track):
    """Parses a connection-profile-file into the ConnectionProfile message."""
    if release_track != base.ReleaseTrack.BETA:
        return util.ParseMessageAndValidateSchema(connection_profile_object_file, 'ConnectionProfile', self._messages.ConnectionProfile)
    data = console_io.ReadFromFileOrStdin(connection_profile_object_file, binary=False)
    try:
        connection_profile_data = yaml.load(data)
    except Exception as e:
        raise ds_exceptions.ParseError('Cannot parse YAML:[{0}]'.format(e))
    display_name = connection_profile_data.get('display_name')
    labels = connection_profile_data.get('labels')
    connection_profile_msg = self._messages.ConnectionProfile(displayName=display_name, labels=labels)
    oracle_profile = self._ParseOracleProfile(connection_profile_data.get('oracle_profile', {}))
    mysql_profile = self._ParseMySqlProfile(connection_profile_data.get('mysql_profile', {}))
    postgresql_profile = self._ParsePostgresqlProfile(connection_profile_data.get('postgresql_profile', {}))
    gcs_profile = self._ParseGCSProfile(connection_profile_data.get('gcs_profile', {}))
    if oracle_profile:
        connection_profile_msg.oracleProfile = oracle_profile
    elif mysql_profile:
        connection_profile_msg.mysqlProfile = mysql_profile
    elif postgresql_profile:
        connection_profile_msg.postgresqlProfile = postgresql_profile
    elif gcs_profile:
        connection_profile_msg.gcsProfile = gcs_profile
    if 'static_service_ip_connectivity' in connection_profile_data:
        connection_profile_msg.staticServiceIpConnectivity = connection_profile_data.get('static_service_ip_connectivity')
    elif 'forward_ssh_connectivity' in connection_profile_data:
        connection_profile_msg.forwardSshConnectivity = connection_profile_data.get('forward_ssh_connectivity')
    elif 'private_connectivity' in connection_profile_data:
        connection_profile_msg.privateConnectivity = connection_profile_data.get('private_connectivity')
    else:
        raise ds_exceptions.ParseError('Cannot parse YAML: missing connectivity method.')
    return connection_profile_msg
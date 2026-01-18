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
def _GetAlloyDBConnectionProfile(self, args, connection_profile_id):
    """Creates an AlloyDB connection profile according to the given args.

    Uses the connection profile ID as the cluster ID, and also sets "postgres"
    as the initial user of the cluster.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.
      connection_profile_id: str, the ID of the connection profile.

    Returns:
      AlloyDBConnectionProfile, to use when creating the connection profile.
    """
    cluster_settings = self.messages.AlloyDbSettings
    primary_settings = self.messages.PrimaryInstanceSettings
    cluster_labels = labels_util.ParseCreateArgs(args, cluster_settings.LabelsValue, 'cluster_labels')
    primary_labels = labels_util.ParseCreateArgs(args, primary_settings.LabelsValue, 'primary_labels')
    database_flags = labels_util.ParseCreateArgs(args, primary_settings.DatabaseFlagsValue, 'database_flags')
    primary_settings = primary_settings(id=args.primary_id, machineConfig=self.messages.MachineConfig(cpuCount=args.cpu_count), databaseFlags=database_flags, labels=primary_labels)
    cluster_settings = cluster_settings(initialUser=self.messages.UserPassword(user='postgres', password=args.password), vpcNetwork=args.network, labels=cluster_labels, primaryInstanceSettings=primary_settings)
    cluster_settings.databaseVersion = self._GetAlloyDBDatabaseVersion(args)
    kms_key_ref = args.CONCEPTS.kms_key.Parse()
    if kms_key_ref is not None:
        cluster_settings.encryptionConfig = self.messages.EncryptionConfig(kmsKeyName=kms_key_ref.RelativeName())
    return self.messages.AlloyDbConnectionProfile(clusterId=connection_profile_id, settings=cluster_settings)
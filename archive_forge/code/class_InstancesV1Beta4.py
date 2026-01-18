from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
class InstancesV1Beta4(_BaseInstances):
    """Common utility functions for sql instances V1Beta4."""

    @staticmethod
    def SetProjectAndInstanceFromRef(instance_resource, instance_ref):
        instance_resource.project = instance_ref.project
        instance_resource.name = instance_ref.instance

    @staticmethod
    def AddBackupConfigToSettings(settings, backup_config):
        settings.backupConfiguration = backup_config

    @staticmethod
    def SetIpConfigurationEnabled(settings, assign_ip):
        settings.ipConfiguration.ipv4Enabled = assign_ip

    @staticmethod
    def SetAuthorizedNetworks(settings, authorized_networks, acl_entry_value):
        settings.ipConfiguration.authorizedNetworks = [acl_entry_value(kind='sql#aclEntry', value=n) for n in authorized_networks]
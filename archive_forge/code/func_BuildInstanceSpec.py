from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
def BuildInstanceSpec(self, name, zone, machine_type, disk_size, preemptible, network, use_with_notebook, source_image=None):
    """Builds an instance spec to be used for Instance creation."""
    disk = self.messages.AttachedDisk(boot=True, autoDelete=True, initializeParams=self.messages.AttachedDiskInitializeParams(sourceImage=source_image, diskSizeGb=disk_size))
    project_number = p_util.GetProjectNumber(properties.VALUES.core.project.Get(required=True))
    network_interface = self.messages.NetworkInterface(network='projects/{}/global/networks/{}'.format(project_number, network), accessConfigs=[self.messages.AccessConfig(name='External NAT', type=self.messages.AccessConfig.TypeValueValuesEnum.ONE_TO_ONE_NAT)])
    metadata = [self.messages.Metadata.ItemsValueListEntry(key='ctpu', value=name)]
    if use_with_notebook:
        metadata.append(self.messages.Metadata.ItemsValueListEntry(key='proxy-mode', value='project_editors'))
    service_account = self.messages.ServiceAccount(email='default', scopes=['https://www.googleapis.com/auth/devstorage.read_write', 'https://www.googleapis.com/auth/logging.write', 'https://www.googleapis.com/auth/monitoring.write', 'https://www.googleapis.com/auth/cloud-platform'])
    labels = self.messages.Instance.LabelsValue(additionalProperties=[self.messages.Instance.LabelsValue.AdditionalProperty(key='ctpu', value=name)])
    return self.messages.Instance(name=name, metadata=self.messages.Metadata(items=metadata), machineType='zones/{}/machineTypes/{}'.format(zone, machine_type), disks=[disk], scheduling=self.messages.Scheduling(preemptible=preemptible), networkInterfaces=[network_interface], labels=labels, serviceAccounts=[service_account])
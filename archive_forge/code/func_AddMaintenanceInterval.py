from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddMaintenanceInterval():
    return base.Argument('--maintenance-interval', type=lambda x: x.upper(), choices={'PERIODIC': 'VMs receive infrastructure and hypervisor updates on a periodic basis, minimizing the number of maintenance operations (live migrations or terminations) on an individual VM. Security updates will still be applied as soon as they are available.', 'RECURRENT': 'VMs receive infrastructure and hypervisor updates on a periodic basis, minimizing the number of maintenance operations (live migrations or terminations) on an individual VM.  This may mean a VM will take longer to receive an update than if it was configured for AS_NEEDED.  Security updates will still be applied as soonas they are available. RECURRENT is used for GEN3 and Sliceof Hardware VMs.'}, help='\n      Specifies how infrastructure upgrades should be applied to the VM.\n      ')
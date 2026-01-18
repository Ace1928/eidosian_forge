from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def CreateLocalNvdimmMessages(args, resources, messages, location=None, scope=None, project=None):
    """Create messages representing local NVDIMMs."""
    local_nvdimms = []
    for local_nvdimm_disk in getattr(args, 'local_nvdimm', []) or []:
        local_nvdimm = _CreateLocalNvdimmMessage(resources, messages, local_nvdimm_disk.get('size'), location, scope, project)
        local_nvdimms.append(local_nvdimm)
    return local_nvdimms
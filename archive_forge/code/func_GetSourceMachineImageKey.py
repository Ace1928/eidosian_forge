from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def GetSourceMachineImageKey(args, source_image, compute_client, holder):
    machine_image_ref = source_image.ResolveAsResource(args, holder.resources)
    csek_keys = csek_utils.CsekKeyStore.FromFile(args.source_machine_image_csek_key_file, allow_rsa_encrypted=False)
    disk_key_or_none = csek_utils.MaybeLookupKeyMessage(csek_keys, machine_image_ref, compute_client.apitools_client)
    return disk_key_or_none
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ValidateIgmReadyForStatefulness(igm_resource, client):
    """Throws exception if IGM is in state not ready for adding statefulness."""
    if not igm_resource.updatePolicy:
        return
    client_update_policy = client.messages.InstanceGroupManagerUpdatePolicy
    type_is_proactive = igm_resource.updatePolicy.type == client_update_policy.TypeValueValuesEnum.PROACTIVE
    replacement_method_is_substitute = igm_resource.updatePolicy.replacementMethod == client_update_policy.ReplacementMethodValueValuesEnum.SUBSTITUTE
    instance_redistribution_type_is_proactive = igm_resource.updatePolicy.instanceRedistributionType == client_update_policy.InstanceRedistributionTypeValueValuesEnum.PROACTIVE
    if type_is_proactive and replacement_method_is_substitute:
        raise exceptions.Error('Stateful IGMs cannot use SUBSTITUTE replacement method. Try `gcloud compute instance-groups managed rolling-update stop-proactive-update')
    if instance_redistribution_type_is_proactive:
        raise exceptions.Error('Stateful regional IGMs cannot use proactive instance redistribution. Try `gcloud compute instance-groups managed update --instance-redistribution-type=NONE')
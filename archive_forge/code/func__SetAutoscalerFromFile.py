from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _SetAutoscalerFromFile(self, autoscaling_file, autoscalers_client, igm_ref, existing_autoscaler_name):
    new_autoscaler = json.loads(files.ReadFileContents(autoscaling_file))
    if new_autoscaler is None:
        if existing_autoscaler_name is None:
            log.info('Configuration specifies no autoscaling and there is no autoscaling configured. Nothing to do.')
            return
        else:
            console_io.PromptContinue(message=_DELETE_AUTOSCALER_PROMPT, cancel_on_no=True, cancel_string=_DELETION_CANCEL_STRING)
            return autoscalers_client.Delete(igm_ref, existing_autoscaler_name)
    new_autoscaler = encoding.DictToMessage(new_autoscaler, autoscalers_client.message_type)
    if existing_autoscaler_name is None:
        managed_instance_groups_utils.AdjustAutoscalerNameForCreation(new_autoscaler, igm_ref)
        return autoscalers_client.Insert(igm_ref, new_autoscaler)
    if getattr(new_autoscaler, 'name', None) and getattr(new_autoscaler, 'name') != existing_autoscaler_name:
        console_io.PromptContinue(message=_REPLACE_AUTOSCALER_PROMPT, cancel_on_no=True, cancel_string=_DELETION_CANCEL_STRING)
        autoscalers_client.Delete(igm_ref, existing_autoscaler_name)
        return autoscalers_client.Insert(igm_ref, new_autoscaler)
    new_autoscaler.name = existing_autoscaler_name
    return autoscalers_client.Update(igm_ref, new_autoscaler)
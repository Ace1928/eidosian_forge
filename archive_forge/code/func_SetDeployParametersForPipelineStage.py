from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def SetDeployParametersForPipelineStage(messages, pipeline_ref, stage):
    """Sets the deployParameter field of cloud deploy delivery pipeline stage message.

  Args:
   messages: module containing the definitions of messages for Cloud Deploy.
   pipeline_ref: protorpc.messages.Message, delivery pipeline resource object.
   stage: dict[str,str], cloud deploy stage yaml definition.
  """
    deploy_parameters = stage.get('deployParameters')
    if deploy_parameters is None:
        return
    _EnsureIsType(deploy_parameters, list, 'failed to parse stages of pipeline {}, deployParameters are defined incorrectly'.format(pipeline_ref.Name()))
    dps_message = getattr(messages, 'DeployParameters')
    dps_values = []
    for dp in deploy_parameters:
        dps_value = dps_message()
        values = dp.get('values')
        if values:
            values_message = dps_message.ValuesValue
            values_dict = values_message()
            _EnsureIsType(values, dict, 'failed to parse stages of pipeline {}, deployParameter values aredefined incorrectly'.format(pipeline_ref.Name()))
            for key, value in values.items():
                values_dict.additionalProperties.append(values_message.AdditionalProperty(key=key, value=value))
            dps_value.values = values_dict
        match_target_labels = dp.get('matchTargetLabels')
        if match_target_labels:
            mtls_message = dps_message.MatchTargetLabelsValue
            mtls_dict = mtls_message()
            for key, value in match_target_labels.items():
                mtls_dict.additionalProperties.append(mtls_message.AdditionalProperty(key=key, value=value))
            dps_value.matchTargetLabels = mtls_dict
        dps_values.append(dps_value)
    stage['deployParameters'] = dps_values
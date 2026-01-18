from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _FillContainerRequirements(container, settings):
    """Set the container CPU and memory limits based on settings."""
    found = set()
    resources = container.resources or RUN_MESSAGES_MODULE.ResourceRequirements()
    limits = resources.limits or RUN_MESSAGES_MODULE.ResourceRequirements.LimitsValue()
    for limit in limits.additionalProperties:
        if limit.key == 'cpu' and settings.cpu:
            limit.value = settings.cpu
        elif limit.key == 'memory' and settings.memory:
            limit.value = settings.memory
        found.add(limit.key)
    if 'cpu' not in found and settings.cpu:
        cpu = RUN_MESSAGES_MODULE.ResourceRequirements.LimitsValue.AdditionalProperty(key='cpu', value=str(settings.cpu))
        limits.additionalProperties.append(cpu)
    if 'memory' not in found and settings.memory:
        mem = RUN_MESSAGES_MODULE.ResourceRequirements.LimitsValue.AdditionalProperty(key='memory', value=str(settings.memory))
        limits.additionalProperties.append(mem)
    resources.limits = limits
    container.resources = resources
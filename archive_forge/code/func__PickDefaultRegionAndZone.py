from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib import init_util
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import network_diagnostics
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _PickDefaultRegionAndZone(self):
    """Pulls metadata properties for region and zone and sets them in gcloud."""
    try:
        project_info = self._RunCmd(['compute', 'project-info', 'describe'], params=['--quiet'])
    except Exception:
        log.status.write('Not setting default zone/region (this feature makes it easier to use\n[gcloud compute] by setting an appropriate default value for the\n--zone and --region flag).\nSee https://cloud.google.com/compute/docs/gcloud-compute section on how to set\ndefault compute region and zone manually. If you would like [gcloud init] to be\nable to do this for you the next time you run it, make sure the\nCompute Engine API is enabled for your project on the\nhttps://console.developers.google.com/apis page.\n\n')
        return None
    default_zone = None
    default_region = None
    if project_info is not None:
        project_info = resource_projector.MakeSerializable(project_info)
        metadata = project_info.get('commonInstanceMetadata', {})
        for item in metadata.get('items', []):
            if item['key'] == 'google-compute-default-zone':
                default_zone = item['value']
            elif item['key'] == 'google-compute-default-region':
                default_region = item['value']
    if not default_zone:
        answer = console_io.PromptContinue(prompt_string='Do you want to configure a default Compute Region and Zone?')
        if not answer:
            return

    def SetProperty(name, default_value, list_command):
        """Set named compute property to default_value or get via list command."""
        if not default_value:
            values = self._RunCmd(list_command)
            if values is None:
                return
            values = list(values)
            message = 'Which Google Compute Engine {0} would you like to use as project default?\nIf you do not specify a {0} via a command line flag while working with Compute Engine resources, the default is assumed.'.format(name)
            idx = console_io.PromptChoice([value['name'] for value in values] + ['Do not set default {0}'.format(name)], message=message, prompt_string=None, allow_freeform=True, freeform_suggester=usage_text.TextChoiceSuggester())
            if idx is None or idx == len(values):
                return
            default_value = values[idx]
        properties.PersistProperty(properties.VALUES.compute.Property(name), default_value['name'])
        log.status.write('Your project default Compute Engine {0} has been set to [{1}].\nYou can change it by running [gcloud config set compute/{0} NAME].\n\n'.format(name, default_value['name']))
        return default_value
    if default_zone:
        default_zone = self._RunCmd(['compute', 'zones', 'describe'], [default_zone])
    zone = SetProperty('zone', default_zone, ['compute', 'zones', 'list'])
    if zone and (not default_region):
        default_region = zone['region']
    if default_region:
        default_region = self._RunCmd(['compute', 'regions', 'describe'], [default_region])
    SetProperty('region', default_region, ['compute', 'regions', 'list'])
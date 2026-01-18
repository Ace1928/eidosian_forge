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
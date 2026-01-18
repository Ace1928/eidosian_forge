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
def _CreateConfiguration(self):
    configuration_name = console_io.PromptResponse("Enter configuration name. Names start with a lower case letter and contain only lower case letters a-z, digits 0-9, and hyphens '-':  ")
    configuration_name = configuration_name.strip()
    named_configs.ConfigurationStore.CreateConfig(configuration_name)
    named_configs.ConfigurationStore.ActivateConfig(configuration_name)
    named_configs.ActivePropertiesFile.Invalidate()
    return configuration_name
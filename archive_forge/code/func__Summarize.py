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
def _Summarize(self, configuration_name):
    log.status.Print('Your Google Cloud SDK is configured and ready to use!\n')
    log.status.Print('* Commands that require authentication will use {0} by default'.format(properties.VALUES.core.account.Get()))
    project = properties.VALUES.core.project.Get()
    if project:
        log.status.Print('* Commands will reference project `{0}` by default'.format(project))
    region = properties.VALUES.compute.region.Get()
    if region:
        log.status.Print('* Compute Engine commands will use region `{0}` by default'.format(region))
    zone = properties.VALUES.compute.zone.Get()
    if zone:
        log.status.Print('* Compute Engine commands will use zone `{0}` by default\n'.format(zone))
    log.status.Print('Run `gcloud help config` to learn how to change individual settings\n')
    log.status.Print('This gcloud configuration is called [{config}]. You can create additional configurations if you work with multiple accounts and/or projects.'.format(config=configuration_name))
    log.status.Print('Run `gcloud topic configurations` to learn more.\n')
    log.status.Print('Some things to try next:\n')
    log.status.Print('* Run `gcloud --help` to see the Cloud Platform services you can interact with. And run `gcloud help COMMAND` to get help on any gcloud command.')
    log.status.Print('* Run `gcloud topic --help` to learn about advanced features of the SDK like arg files and output formatting')
    log.status.Print('* Run `gcloud cheat-sheet` to see a roster of go-to `gcloud` commands.')
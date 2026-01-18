from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import exceptions as api_lib_util_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.projects import util as command_lib_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
def WarnIfSettingApiEndpointOverrideOutsideOfConfigUniverse(value, prop):
    """Warn if setting 'api_endpoint_overrides/<api>' outside universe_domain."""
    universe_domain = properties.VALUES.core.universe_domain.Get()
    if universe_domain not in value:
        log.warning(f'The value set for [{prop}] was found to be associated with a universe domain outside of the current config universe [{universe_domain}]. Please create a new gcloud configuration for each universe domain you make requests to using `gcloud config configurations create` with the `--universe-domain` flag or switch to a configuration associated with [{value}].')
        return True
    return False
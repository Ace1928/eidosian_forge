from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import build as app_build
from googlecloudsdk.api_lib.app import cloud_build
from googlecloudsdk.api_lib.app import docker_image
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.api_lib.cloudbuild import build as cloudbuild_build
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as s_exceptions
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.tools import context_util
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
def UseSsl(service_info):
    """Returns whether the root URL for an application is served over HTTPS.

  More specifically, returns the 'secure' setting of the handler that will serve
  the application. This can be 'always', 'optional', or 'never', depending on
  when the URL is served over HTTPS.

  Will miss a small number of cases, but HTTP is always okay (an HTTP URL to an
  HTTPS-only service will result in a redirect).

  Args:
    service_info: ServiceYamlInfo, the service configuration.

  Returns:
    str, the 'secure' setting of the handler for the root URL.
  """
    if service_info.is_ti_runtime and (not service_info.parsed.handlers):
        return appinfo.SECURE_HTTP_OR_HTTPS
    for handler in service_info.parsed.handlers:
        try:
            if re.match(handler.url + '$', '/'):
                return handler.secure
        except re.error:
            pass
    return appinfo.SECURE_HTTP
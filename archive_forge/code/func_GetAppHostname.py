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
def GetAppHostname(app=None, app_id=None, service=None, version=None, use_ssl=appinfo.SECURE_HTTP, deploy=True):
    """Returns the hostname of the given version of the deployed app.

  Args:
    app: Application resource. One of {app, app_id} must be given.
    app_id: str, project ID. One of {app, app_id} must be given. If both are
      provided, the hostname from app is preferred.
    service: str, the (optional) service being deployed
    version: str, the deployed version ID (omit to get the default version URL).
    use_ssl: bool, whether to construct an HTTPS URL.
    deploy: bool, if this is called during a deployment.

  Returns:
    str. Constructed URL.

  Raises:
    TypeError: if neither an app nor an app_id is provided
  """
    if not app and (not app_id):
        raise TypeError('Must provide an application resource or application ID.')
    version = version or ''
    service_name = service or ''
    if service == DEFAULT_SERVICE:
        service_name = ''
    if not app:
        api_client = appengine_api_client.AppengineApiClient.GetApiClient()
        app = api_client.GetApplication()
    if app:
        app_id, domain = app.defaultHostname.split('.', 1)
    subdomain_parts = list(filter(bool, [version, service_name, app_id]))
    scheme = 'http'
    if use_ssl == appinfo.SECURE_HTTP:
        subdomain = '.'.join(subdomain_parts)
        scheme = 'http'
    else:
        subdomain = ALT_SEPARATOR.join(subdomain_parts)
        if len(subdomain) <= MAX_DNS_LABEL_LENGTH:
            scheme = 'https'
        else:
            if deploy:
                format_parts = ['$VERSION_ID', '$SERVICE_ID', '$APP_ID']
                subdomain_format = ALT_SEPARATOR.join([j for i, j in zip([version, service_name, app_id], format_parts) if i])
                msg = 'This deployment will result in an invalid SSL certificate for service [{0}]. The total length of your subdomain in the format {1} should not exceed {2} characters. Please verify that the certificate corresponds to the parent domain of your application when you connect.'.format(service, subdomain_format, MAX_DNS_LABEL_LENGTH)
                log.warning(msg)
            subdomain = '.'.join(subdomain_parts)
            if use_ssl == appinfo.SECURE_HTTP_OR_HTTPS:
                scheme = 'http'
            elif use_ssl == appinfo.SECURE_HTTPS:
                if not deploy:
                    msg = 'Most browsers will reject the SSL certificate for service [{0}].'.format(service)
                    log.warning(msg)
                scheme = 'https'
    return '{0}://{1}.{2}'.format(scheme, subdomain, domain)
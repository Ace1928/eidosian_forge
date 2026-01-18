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
class WindowMaxPathError(Error):
    """Raised if a file cannot be read because of the MAX_PATH limitation."""
    _WINDOWS_MAX_PATH_ERROR_TEMPLATE = "The following file couldn't be read because its path is too long:\n\n  [{0}]\n\nFor more information on this issue and possible workarounds, please read the\nfollowing (links are specific to Node.js, but the information is generally\napplicable):\n\n* https://github.com/Microsoft/nodejstools/issues/69\n* https://github.com/Microsoft/nodejs-guidelines/blob/master/windows-environment.md#max_path-explanation-and-workarounds"

    def __init__(self, filename):
        super(WindowMaxPathError, self).__init__(self._WINDOWS_MAX_PATH_ERROR_TEMPLATE.format(filename))
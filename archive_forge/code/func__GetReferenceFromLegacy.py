from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def _GetReferenceFromLegacy(self):
    """Tries to resolve the reference by the legacy resolution process.

    TODO(b/37542861): This can be removed after all runtimes have been migrated
    to publish their builders in the manifest instead of <runtime>.version
    files.

    If the runtime is not found in the manifest, use legacy resolution. If the
    app.yaml contains a runtime_config.runtime_version, this loads the file from
    '<runtime>-<version>.yaml' in the runtime builders root. Otherwise, it
    checks '<runtime>.version' to get the default version, and loads the
    configuration for that version.

    Returns:
      BuilderReference or None
    """
    if self.legacy_runtime_version:
        return self._GetReferenceFromLegacyWithVersion(self.legacy_runtime_version)
    log.debug('Fetching version for runtime [%s] in legacy mode', self.runtime)
    version_file_name = self.runtime + '.version'
    version_file_uri = _Join(self.build_file_root, version_file_name)
    try:
        with _Read(version_file_uri) as f:
            version = f.read().decode().strip()
    except FileReadError:
        log.debug('', exc_info=True)
        return None
    log.debug('Using version [%s] for runtime [%s] in legacy mode', version, self.runtime)
    return self._GetReferenceFromLegacyWithVersion(version)
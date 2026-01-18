from __future__ import absolute_import, division, print_function
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
def _service_profile_enabled(self, service):
    """Returns `True` if the service has no profiles defined or has a profile which is among
           the profiles passed to the `docker compose up` command. Otherwise returns `False`.
        """
    if LooseVersion(compose_version) < LooseVersion('1.28.0'):
        return True
    return service.enabled_for_profiles(self.profiles or [])
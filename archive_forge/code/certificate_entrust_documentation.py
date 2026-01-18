from __future__ import absolute_import, division, print_function
import datetime
import time
import os
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import ECSClient, RestOperationException, SessionConfigurationException
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
Return bytes for self.cert.
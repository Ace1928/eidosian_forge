from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import six
from gslib.utils.unit_util import ONE_GIB
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
class Scopes(object):
    """Enum class for auth scopes, as unicode."""
    CLOUD_PLATFORM = 'https://www.googleapis.com/auth/cloud-platform'
    CLOUD_PLATFORM_READ_ONLY = 'https://www.googleapis.com/auth/cloud-platform.read-only'
    FULL_CONTROL = 'https://www.googleapis.com/auth/devstorage.full_control'
    READ_ONLY = 'https://www.googleapis.com/auth/devstorage.read_only'
    READ_WRITE = 'https://www.googleapis.com/auth/devstorage.read_write'
    REAUTH = 'https://www.googleapis.com/auth/accounts.reauth'
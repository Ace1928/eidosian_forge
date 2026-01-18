from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
from six.moves import urllib
def DockerFunc(*args, **kwargs):
    try:
        return func(*args, **kwargs)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise DockerError(DOCKER_NOT_FOUND_ERROR)
        else:
            raise
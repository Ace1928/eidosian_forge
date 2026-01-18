from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import enum
import json
import os
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import _mtls_helper
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _GetPlatform():
    platform = platforms.Platform.Current()
    if platform.operating_system == platforms.OperatingSystem.MACOSX and platform.architecture == platforms.Architecture.x86_64:
        if platforms.Platform.IsActuallyM1ArmArchitecture():
            platform.architecture = platforms.Architecture.arm
    return platform
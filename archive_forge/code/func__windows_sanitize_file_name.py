from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def _windows_sanitize_file_name(file_name):
    """Converts colons and characters that make Windows upset."""
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS and properties.VALUES.storage.convert_incompatible_windows_path_characters.GetBool():
        return platforms.MakePathWindowsCompatible(file_name)
    return file_name
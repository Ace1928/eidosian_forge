import os
import re
import sys
def _supports_arm64_builds():
    """Returns True if arm64 builds are supported on this system"""
    osx_version = _get_system_version_tuple()
    return osx_version >= (11, 0) if osx_version else False
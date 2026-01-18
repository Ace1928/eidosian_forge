import os
import re
import sys
def _supports_universal_builds():
    """Returns True if universal builds are supported on this system"""
    osx_version = _get_system_version_tuple()
    return bool(osx_version >= (10, 4)) if osx_version else False
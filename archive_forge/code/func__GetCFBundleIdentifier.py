import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def _GetCFBundleIdentifier(self):
    """Extracts CFBundleIdentifier value from Info.plist in the bundle.

    Returns:
      Value of CFBundleIdentifier in the Info.plist located in the bundle.
    """
    info_plist_path = os.path.join(os.environ['TARGET_BUILD_DIR'], os.environ['INFOPLIST_PATH'])
    info_plist_data = self._LoadPlistMaybeBinary(info_plist_path)
    return info_plist_data['CFBundleIdentifier']
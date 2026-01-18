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
def _LoadProvisioningProfile(self, profile_path):
    """Extracts the plist embedded in a provisioning profile.

    Args:
      profile_path: string, path to the .mobileprovision file

    Returns:
      Content of the plist embedded in the provisioning profile as a dictionary.
    """
    with tempfile.NamedTemporaryFile() as temp:
        subprocess.check_call(['security', 'cms', '-D', '-i', profile_path, '-o', temp.name])
        return self._LoadPlistMaybeBinary(temp.name)
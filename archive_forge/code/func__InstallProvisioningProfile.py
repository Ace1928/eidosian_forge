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
def _InstallProvisioningProfile(self, profile, bundle_identifier):
    """Installs embedded.mobileprovision into the bundle.

    Args:
      profile: string, optional, short name of the .mobileprovision file
        to use, if empty or the file is missing, the best file installed
        will be used
      bundle_identifier: string, value of CFBundleIdentifier from Info.plist

    Returns:
      A tuple containing two dictionary: variables substitutions and values
      to overrides when generating the entitlements file.
    """
    source_path, provisioning_data, team_id = self._FindProvisioningProfile(profile, bundle_identifier)
    target_path = os.path.join(os.environ['BUILT_PRODUCTS_DIR'], os.environ['CONTENTS_FOLDER_PATH'], 'embedded.mobileprovision')
    shutil.copy2(source_path, target_path)
    substitutions = self._GetSubstitutions(bundle_identifier, team_id + '.')
    return (substitutions, provisioning_data['Entitlements'])
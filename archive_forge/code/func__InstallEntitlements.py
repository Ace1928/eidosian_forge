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
def _InstallEntitlements(self, entitlements, substitutions, overrides):
    """Generates and install the ${BundleName}.xcent entitlements file.

    Expands variables "$(variable)" pattern in the source entitlements file,
    add extra entitlements defined in the .mobileprovision file and the copy
    the generated plist to "${BundlePath}.xcent".

    Args:
      entitlements: string, optional, path to the Entitlements.plist template
        to use, defaults to "${SDKROOT}/Entitlements.plist"
      substitutions: dictionary, variable substitutions
      overrides: dictionary, values to add to the entitlements

    Returns:
      Path to the generated entitlements file.
    """
    source_path = entitlements
    target_path = os.path.join(os.environ['BUILT_PRODUCTS_DIR'], os.environ['PRODUCT_NAME'] + '.xcent')
    if not source_path:
        source_path = os.path.join(os.environ['SDKROOT'], 'Entitlements.plist')
    shutil.copy2(source_path, target_path)
    data = self._LoadPlistMaybeBinary(target_path)
    data = self._ExpandVariables(data, substitutions)
    if overrides:
        for key in overrides:
            if key not in data:
                data[key] = overrides[key]
    plistlib.writePlist(data, target_path)
    return target_path
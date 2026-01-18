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
def _WritePkgInfo(self, info_plist):
    """This writes the PkgInfo file from the data stored in Info.plist."""
    plist = plistlib.readPlist(info_plist)
    if not plist:
        return
    package_type = plist['CFBundlePackageType']
    if package_type != 'APPL':
        return
    signature_code = plist.get('CFBundleSignature', '????')
    if len(signature_code) != 4:
        signature_code = '?' * 4
    dest = os.path.join(os.path.dirname(info_plist), 'PkgInfo')
    with open(dest, 'w') as fp:
        fp.write(f'{package_type}{signature_code}')
import collections
import os
import re
import sys
import functools
import itertools
def _mac_ver_xml():
    fn = '/System/Library/CoreServices/SystemVersion.plist'
    if not os.path.exists(fn):
        return None
    try:
        import plistlib
    except ImportError:
        return None
    with open(fn, 'rb') as f:
        pl = plistlib.load(f)
    release = pl['ProductVersion']
    versioninfo = ('', '', '')
    machine = os.uname().machine
    if machine in ('ppc', 'Power Macintosh'):
        machine = 'PowerPC'
    return (release, versioninfo, machine)
from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _getmember(self, name, tarinfo=None, normalize=False):
    """Find an archive member by name from bottom to top.
           If tarinfo is given, it is used as the starting point.
        """
    members = self.getmembers()
    skipping = False
    if tarinfo is not None:
        try:
            index = members.index(tarinfo)
        except ValueError:
            skipping = True
        else:
            members = members[:index]
    if normalize:
        name = os.path.normpath(name)
    for member in reversed(members):
        if skipping:
            if tarinfo.offset == member.offset:
                skipping = False
            continue
        if normalize:
            member_name = os.path.normpath(member.name)
        else:
            member_name = member.name
        if name == member_name:
            return member
    if skipping:
        raise ValueError(tarinfo)
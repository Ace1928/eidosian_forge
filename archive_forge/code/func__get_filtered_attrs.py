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
def _get_filtered_attrs(member, dest_path, for_data=True):
    new_attrs = {}
    name = member.name
    dest_path = os.path.realpath(dest_path)
    if name.startswith(('/', os.sep)):
        name = new_attrs['name'] = member.path.lstrip('/' + os.sep)
    if os.path.isabs(name):
        raise AbsolutePathError(member)
    target_path = os.path.realpath(os.path.join(dest_path, name))
    if os.path.commonpath([target_path, dest_path]) != dest_path:
        raise OutsideDestinationError(member, target_path)
    mode = member.mode
    if mode is not None:
        mode = mode & 493
        if for_data:
            if member.isreg() or member.islnk():
                if not mode & 64:
                    mode &= ~73
                mode |= 384
            elif member.isdir() or member.issym():
                mode = None
            else:
                raise SpecialFileError(member)
        if mode != member.mode:
            new_attrs['mode'] = mode
    if for_data:
        if member.uid is not None:
            new_attrs['uid'] = None
        if member.gid is not None:
            new_attrs['gid'] = None
        if member.uname is not None:
            new_attrs['uname'] = None
        if member.gname is not None:
            new_attrs['gname'] = None
        if member.islnk() or member.issym():
            if os.path.isabs(member.linkname):
                raise AbsoluteLinkError(member)
            if member.issym():
                target_path = os.path.join(dest_path, os.path.dirname(name), member.linkname)
            else:
                target_path = os.path.join(dest_path, member.linkname)
            target_path = os.path.realpath(target_path)
            if os.path.commonpath([target_path, dest_path]) != dest_path:
                raise LinkOutsideDestinationError(member, target_path)
    return new_attrs
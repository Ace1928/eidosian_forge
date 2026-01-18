import errno
import glob
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import os.path
import struct
import sys
def _build_cacheable_data():
    real_groups = importlib_metadata.entry_points()
    if not isinstance(real_groups, dict):
        real_groups = {group: real_groups.select(group=group) for group in real_groups.groups}
    groups = {}
    for name, group_data in real_groups.items():
        existing = set()
        members = []
        groups[name] = members
        for ep in group_data:
            item = (ep.name, ep.value, ep.group)
            if item in existing:
                continue
            existing.add(item)
            members.append(item)
    return {'groups': groups, 'sys.executable': sys.executable, 'sys.prefix': sys.prefix}
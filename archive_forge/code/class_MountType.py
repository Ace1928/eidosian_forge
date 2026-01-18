from __future__ import annotations
import codecs
import dataclasses
import pathlib
import re
class MountType:
    """Linux filesystem mount type constants."""
    TMPFS = 'tmpfs'
    CGROUP_V1 = 'cgroup'
    CGROUP_V2 = 'cgroup2'
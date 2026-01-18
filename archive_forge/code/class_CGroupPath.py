from __future__ import annotations
import codecs
import dataclasses
import pathlib
import re
class CGroupPath:
    """Linux cgroup path constants."""
    ROOT = '/sys/fs/cgroup'
    SYSTEMD = '/sys/fs/cgroup/systemd'
    SYSTEMD_RELEASE_AGENT = '/sys/fs/cgroup/systemd/release_agent'
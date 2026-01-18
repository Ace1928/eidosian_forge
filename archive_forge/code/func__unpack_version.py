from __future__ import annotations
from difflib import SequenceMatcher
from typing import Iterable, Iterator
from kombu import version_info_t
def _unpack_version(major: str, minor: str | int=0, micro: str | int=0, releaselevel: str='', serial: str='') -> version_info_t:
    return version_info_t(int(major), int(minor), micro, releaselevel, serial)
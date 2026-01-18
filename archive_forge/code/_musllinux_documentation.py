import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Optional
from ._elffile import ELFFile
Generate musllinux tags compatible to the current platform.

    :param arch: Should be the part of platform tag after the ``linux_``
        prefix, e.g. ``x86_64``. The ``linux_`` prefix is assumed as a
        prerequisite for the current platform to be musllinux-compatible.

    :returns: An iterator of compatible musllinux tags.
    
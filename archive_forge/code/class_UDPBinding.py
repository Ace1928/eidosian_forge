from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
@attrs.frozen
class UDPBinding:
    local: UDPEndpoint
from base64 import b64encode
from typing import Mapping, Optional, NamedTuple
import logging
import pkg_resources
from cloudsdk.google.protobuf import struct_pb2  # pytype: disable=pyi-error
class _Semver(NamedTuple):
    major: int
    minor: int
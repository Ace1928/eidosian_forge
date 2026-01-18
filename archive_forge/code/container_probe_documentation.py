from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import pwd
import typing as t
from ..io import (
from ..util import (
from ..config import (
from ..docker_util import (
from ..host_configs import (
from ..cgroup import (
Generate and return an identity string to use when logging test results.
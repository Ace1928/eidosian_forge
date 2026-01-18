import contextlib
import hashlib
import logging
import os
from types import TracebackType
from typing import Dict, Generator, Optional, Set, Type, Union
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.temp_dir import TempDirectory
Ensure that `key` cannot install itself as a setup requirement.

        :raises LookupError: If `key` was already provided in a parent invocation of
                             the context introduced by this method.
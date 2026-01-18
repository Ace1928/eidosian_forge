import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
class MessageScope(enum.Enum):
    """Scopes for message listeners"""
    ALL = 'all'
    CLIENT = 'client'
    SERVER = 'server'
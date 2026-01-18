import os
import pathlib
import uuid
from typing import Any, Dict, List, NewType, Optional, Union, cast
from dataclasses import dataclass, fields
from jupyter_core.utils import ensure_async
from tornado import web
from traitlets import Instance, TraitError, Unicode, validate
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.traittypes import InstanceFromClasses
class KernelSessionRecordConflict(Exception):
    """Exception class to use when two KernelSessionRecords cannot
    merge because of conflicting data.
    """
from __future__ import annotations
import contextlib
import inspect
import os
import sys
import threading
import time
import uuid
from collections.abc import Sized
from functools import wraps
from typing import Any, Callable, Final, TypeVar, cast, overload
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Argument, Command
def _get_machine_id_v3() -> str:
    """Get the machine ID

    This is a unique identifier for a user for tracking metrics in Segment,
    that is broken in different ways in some Linux distros and Docker images.
    - at times just a hash of '', which means many machines map to the same ID
    - at times a hash of the same string, when running in a Docker container
    """
    machine_id = str(uuid.getnode())
    if os.path.isfile(_ETC_MACHINE_ID_PATH):
        with open(_ETC_MACHINE_ID_PATH) as f:
            machine_id = f.read()
    elif os.path.isfile(_DBUS_MACHINE_ID_PATH):
        with open(_DBUS_MACHINE_ID_PATH) as f:
            machine_id = f.read()
    return machine_id
import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union
def _try_parse_port(port_str: str) -> Optional[int]:
    """Try to extract the port number from ``port_str``."""
    if port_str and re.match('^[0-9]{1,5}$', port_str):
        return int(port_str)
    return None
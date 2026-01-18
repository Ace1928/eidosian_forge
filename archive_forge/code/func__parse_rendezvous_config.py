import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union
def _parse_rendezvous_config(config_str: str) -> Dict[str, str]:
    """Extract key-value pairs from a rendezvous configuration string.

    Args:
        config_str:
            A string in format <key1>=<value1>,...,<keyN>=<valueN>.
    """
    config: Dict[str, str] = {}
    config_str = config_str.strip()
    if not config_str:
        return config
    key_values = config_str.split(',')
    for kv in key_values:
        key, *values = kv.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError('The rendezvous configuration string must be in format <key1>=<value1>,...,<keyN>=<valueN>.')
        value: Optional[str]
        if values:
            value = values[0].strip()
        else:
            value = None
        if not value:
            raise ValueError(f"The rendezvous configuration option '{key}' must have a value specified.")
        config[key] = value
    return config
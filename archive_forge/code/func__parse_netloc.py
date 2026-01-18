import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union
from wandb.sdk.launch.errors import LaunchError
def _parse_netloc(netloc: str) -> Tuple[Optional[str], Optional[str], str]:
    """Parse netloc into username, password, and host.

    github.com => None, None, "@github.com"
    username@github.com => "username", None, "github.com"
    username:password@github.com => "username", "password", "github.com"
    """
    parts = netloc.split('@', 1)
    if len(parts) == 1:
        return (None, None, parts[0])
    auth, host = parts
    parts = auth.split(':', 1)
    if len(parts) == 1:
        return (parts[0], None, host)
    return (parts[0], parts[1], host)
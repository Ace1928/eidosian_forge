from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from huggingface_hub.utils import parse_datetime
class SpaceStage(str, Enum):
    """
    Enumeration of possible stage of a Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceStage.BUILDING == "BUILDING"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceInfo.ts#L61 (private url).
    """
    NO_APP_FILE = 'NO_APP_FILE'
    CONFIG_ERROR = 'CONFIG_ERROR'
    BUILDING = 'BUILDING'
    BUILD_ERROR = 'BUILD_ERROR'
    RUNNING = 'RUNNING'
    RUNNING_BUILDING = 'RUNNING_BUILDING'
    RUNTIME_ERROR = 'RUNTIME_ERROR'
    DELETING = 'DELETING'
    STOPPED = 'STOPPED'
    PAUSED = 'PAUSED'
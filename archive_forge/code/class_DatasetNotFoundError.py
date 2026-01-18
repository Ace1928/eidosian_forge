from typing import Any, Dict, List, Optional, Union
from huggingface_hub import HfFileSystem
from . import config
from .table import CastError
from .utils.track import TrackedIterable, tracked_list, tracked_str
class DatasetNotFoundError(FileNotFoundDatasetsError):
    """Dataset not found.

    Raised when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    """
from typing import Any, Dict, List, Optional, Union
from .. import config
from ..exceptions import DatasetsError
from .file_utils import (
from .logging import get_logger

    Get the dataset information, can be useful to get e.g. the dataset features.
    Docs: https://huggingface.co/docs/datasets-server/info
    
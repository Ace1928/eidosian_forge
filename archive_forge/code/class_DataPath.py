import typing
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, List, Literal, Optional, Tuple, Union
from .params import Description, ParamArg, ParamVal
class DataPath(PurePosixPath):
    """Type for Dataset Path, relative to the foldermap.json file."""

    def __repr__(self) -> str:
        return repr(str(self))
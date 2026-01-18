import asyncio
import os
from typing import Awaitable, Tuple, Type, TypeVar, Union
from typing import Dict as TypeDict
from typing import List as TypeList
from pathlib import Path
from traitlets.traitlets import Dict, Float, List, default
from nbclient.util import ensure_async
import re
from .notebook_renderer import NotebookRenderer
from .utils import ENV_VARIABLE
class NormalKernelManager(base_class):

    @property
    def notebook_data(self) -> TypeDict:
        return {}

    def get_pool_size(self, nb: str) -> int:
        return 0
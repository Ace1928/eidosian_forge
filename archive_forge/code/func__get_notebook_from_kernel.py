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
def _get_notebook_from_kernel(self, kernel_id: str) -> Union[None, str]:
    """Helper to get notebook name from heated kernel id.

                Args:
                    kernel_id (str): Kernel id

                Returns:
                    Union[None, str]: return associated notebook with kernel id.

                """
    for nb_name, data in self.notebook_data.items():
        if kernel_id in data['kernel_ids']:
            return nb_name
    return None
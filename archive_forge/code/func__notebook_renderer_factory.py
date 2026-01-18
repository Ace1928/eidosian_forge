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
def _notebook_renderer_factory(self, notebook_path: Union[str, None]=None) -> NotebookRenderer:
    """Helper function to create `NotebookRenderer` instance.

                Args:
                    - notebook_path (Union[str, None], optional): Path to the
                    notebook. Defaults to None.
                """
    return NotebookRenderer(voila_configuration=self.parent.voila_configuration, traitlet_config=self.parent.config, notebook_path=notebook_path, template_paths=self.parent.template_paths, config_manager=self.parent.config_manager, contents_manager=self.parent.contents_manager, base_url=self.parent.base_url, kernel_spec_manager=self.parent.kernel_spec_manager)
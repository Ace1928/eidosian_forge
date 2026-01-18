import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
from .agent_types import handle_agent_inputs, handle_agent_outputs
from {module_name} import {class_name}
class GradioToolWrapper(Tool):

    def __init__(self, _gradio_tool):
        super().__init__()
        self.name = _gradio_tool.name
        self.description = _gradio_tool.description
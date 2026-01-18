from __future__ import annotations
import warnings
from typing import (
import numpy as np
import semantic_version
from gradio_client.documentation import document
from gradio.components import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
class DataframeData(GradioModel):
    headers: List[str]
    data: Union[List[List[Any]], List[Tuple[Any, ...]]]
    metadata: Optional[Dict[str, Optional[List[Any]]]] = None
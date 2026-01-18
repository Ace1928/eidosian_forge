from __future__ import annotations
import json
import operator
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
class LabelData(GradioModel):
    label: Optional[Union[str, int, float]] = None
    confidences: Optional[List[LabelConfidence]] = None
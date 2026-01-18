from __future__ import annotations
from typing import Any, Callable, List, Union
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import Events
class HighlightedToken(GradioModel):
    token: str
    class_or_confidence: Union[str, float, None] = None
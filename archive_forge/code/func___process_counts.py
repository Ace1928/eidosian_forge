from __future__ import annotations
import warnings
from typing import (
import numpy as np
import semantic_version
from gradio_client.documentation import document
from gradio.components import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
@staticmethod
def __process_counts(count, default=3) -> tuple[int, str]:
    if count is None:
        return (default, 'dynamic')
    if isinstance(count, (int, float)):
        return (int(count), 'dynamic')
    else:
        return count
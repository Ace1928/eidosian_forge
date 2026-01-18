from __future__ import annotations
import copy
import json
from typing import Any, Literal
from gradio_client.documentation import document
from gradio.components import Button, Component
from gradio.context import Context
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.utils import resolve_singleton

        Parameters:
            value: string corresponding to the button label
        Returns:
            Expects a `str` value that is set as the button label
        
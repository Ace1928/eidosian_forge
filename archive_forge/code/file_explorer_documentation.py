from __future__ import annotations
import fnmatch
import os
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal
from gradio_client.documentation import document
from gradio.components.base import Component, server
from gradio.data_classes import GradioRootModel

        Returns:
            a list of dictionaries, where each dictionary represents a file or subdirectory in the given subdirectory
        
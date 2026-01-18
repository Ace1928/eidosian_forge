from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import Events

        Parameters:
            value: expects a `list[list[str | None | tuple]]`, i.e. a list of lists. The inner list should have 2 elements: the user message and the response message. The individual messages can be (1) strings in valid Markdown, (2) tuples if sending files: (a filepath or URL to a file, [optional string alt text]) -- if the file is image/video/audio, it is displayed in the Chatbot, or (3) None, in which case the message is not displayed.
        Returns:
            an object of type ChatbotData
        
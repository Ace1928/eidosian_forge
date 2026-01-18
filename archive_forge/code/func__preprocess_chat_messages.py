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
def _preprocess_chat_messages(self, chat_message: str | FileMessage | None) -> str | tuple[str | None] | tuple[str | None, str] | None:
    if chat_message is None:
        return None
    elif isinstance(chat_message, FileMessage):
        if chat_message.alt_text is not None:
            return (chat_message.file.path, chat_message.alt_text)
        else:
            return (chat_message.file.path,)
    elif isinstance(chat_message, str):
        return chat_message
    else:
        raise ValueError(f'Invalid message for Chatbot component: {chat_message}')
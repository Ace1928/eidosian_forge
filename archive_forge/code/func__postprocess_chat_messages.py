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
def _postprocess_chat_messages(self, chat_message: str | tuple | list | None) -> str | FileMessage | None:
    if chat_message is None:
        return None
    elif isinstance(chat_message, (tuple, list)):
        filepath = str(chat_message[0])
        mime_type = client_utils.get_mimetype(filepath)
        return FileMessage(file=FileData(path=filepath, mime_type=mime_type), alt_text=chat_message[1] if len(chat_message) > 1 else None)
    elif isinstance(chat_message, str):
        chat_message = inspect.cleandoc(chat_message)
        return chat_message
    else:
        raise ValueError(f'Invalid message for Chatbot component: {chat_message}')
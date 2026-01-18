from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import (
import param
from ..io.resources import CDN_DIST
from ..layout import Row, Tabs
from ..pane.image import ImageBase
from ..viewable import Viewable
from ..widgets.base import Widget
from ..widgets.button import Button
from ..widgets.input import FileInput, TextInput
from .feed import CallbackState, ChatFeed
from .input import ChatAreaInput
from .message import ChatMessage, _FileInputMessage
def _serialize_for_transformers(self, messages: List[ChatMessage], role_names: Dict[str, str | List[str]] | None=None, default_role: str | None='assistant', custom_serializer: Callable=None) -> List[Dict[str, Any]]:
    """
        Exports the chat log for use with transformers.

        Arguments
        ---------
        messages : list(ChatMessage)
            A list of ChatMessage objects to serialize.
        role_names : dict(str, str | list(str)) | None
            A dictionary mapping the role to the ChatMessage's user name.
            Defaults to `{"user": [self.user], "assistant": [self.callback_user]}`
            if not set. The keys and values are case insensitive as the strings
            will all be lowercased. The values can be a string or a list of strings,
            e.g. `{"user": "user", "assistant": ["executor", "langchain"]}`.
        default_role : str
            The default role to use if the user name is not found in role_names.
            If this is set to None, raises a ValueError if the user name is not found.
        custom_serializer : callable
            A custom function to format the ChatMessage's object. The function must
            accept one positional argument, the ChatMessage object, and return a string.
            If not provided, uses the serialize method on ChatMessage.

        Returns
        -------
        A list of dictionaries with a role and content keys.
        """
    if role_names is None:
        role_names = {'user': [self.user], 'assistant': [self.callback_user]}
    return super()._serialize_for_transformers(messages, role_names, default_role, custom_serializer)
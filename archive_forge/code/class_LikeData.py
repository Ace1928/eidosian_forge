from __future__ import annotations
import dataclasses
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence
from gradio_client.documentation import document
from jinja2 import Template
from gradio.context import Context
from gradio.utils import get_cancel_function
class LikeData(EventData):

    def __init__(self, target: Block | None, data: Any):
        super().__init__(target, data)
        self.index: int | tuple[int, int] = data['index']
        '\n        The index of the liked/disliked item. Is a tuple if the component is two dimensional.\n        '
        self.value: Any = data['value']
        '\n        The value of the liked/disliked item.\n        '
        self.liked: bool = data.get('liked', True)
        '\n        True if the item was liked, False if disliked.\n        '
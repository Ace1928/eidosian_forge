from __future__ import annotations
import dataclasses
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence
from gradio_client.documentation import document
from jinja2 import Template
from gradio.context import Context
from gradio.utils import get_cancel_function
class KeyUpData(EventData):

    def __init__(self, target: Block | None, data: Any):
        super().__init__(target, data)
        self.key: str = data['key']
        '\n        The key that was pressed.\n        '
        self.input_value: str = data['input_value']
        '\n        The displayed value in the input textbox after the key was pressed. This may be different than the `value`\n        attribute of the component itself, as the `value` attribute of some components (e.g. Dropdown) are not updated\n        until the user presses Enter.\n        '
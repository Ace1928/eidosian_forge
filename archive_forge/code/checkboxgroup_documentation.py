from __future__ import annotations
from typing import Any, Callable, Literal
from gradio_client.documentation import document
from gradio.components.base import FormComponent
from gradio.events import Events

        Parameters:
            value: Expects a `list[str | int | float]` of values or a single `str | int | float` value, the checkboxes with these values are checked.
        Returns:
            the list of checked checkboxes' values
        
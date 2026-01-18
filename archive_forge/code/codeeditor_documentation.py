from __future__ import annotations
from typing import (
import param
from pyviz_comms import JupyterComm
from ..models.enums import ace_themes
from ..util import lazy_load
from .base import Widget

    The CodeEditor widget allows displaying and editing code in the
    powerful Ace editor.

    Reference: https://panel.holoviz.org/reference/widgets/CodeEditor.html

    :Example:

    >>> CodeEditor(value=py_code, language='python', theme='monokai')
    
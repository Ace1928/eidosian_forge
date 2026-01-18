from __future__ import annotations
import dataclasses
import inspect
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Literal
import gradio
import gradio as gr
from {package_name} import {name}
import gradio as gr
from {package_name} import {name}
from .{name.lower()} import {name}
@dataclasses.dataclass
class ComponentFiles:
    template: str
    demo_code: str = default_demo_code
    python_file_name: str = ''
    js_dir: str = ''

    def __post_init__(self):
        self.js_dir = self.js_dir or self.template.lower()
        self.python_file_name = self.python_file_name or f'{self.template.lower()}.py'
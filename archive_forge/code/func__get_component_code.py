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
def _get_component_code(template: str | None) -> ComponentFiles:
    template = template or 'Fallback'
    if template in OVERRIDES:
        return OVERRIDES[template]
    else:
        return ComponentFiles(python_file_name=f'{template.lower()}.py', js_dir=template.lower(), template=template)
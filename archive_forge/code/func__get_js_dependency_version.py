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
def _get_js_dependency_version(name: str, local_js_dir: Path) -> str:
    package_json = json.loads(Path(local_js_dir / name.split('/')[1] / 'package.json').read_text())
    return package_json['version']
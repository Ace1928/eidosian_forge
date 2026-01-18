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
def _modify_js_deps(package_json: dict, key: Literal['dependencies', 'devDependencies'], gradio_dir: Path):
    for dep in package_json.get(key, []):
        if not _in_test_dir() and dep.startswith('@gradio/'):
            package_json[key][dep] = _get_js_dependency_version(dep, gradio_dir / '_frontend_code')
    return package_json
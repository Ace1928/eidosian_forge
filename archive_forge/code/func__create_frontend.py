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
def _create_frontend(name: str, component: ComponentFiles, directory: Path, package_name: str):
    frontend = directory / 'frontend'
    frontend.mkdir(exist_ok=True)
    p = Path(inspect.getfile(gradio)).parent

    def ignore(_src, names):
        ignored = []
        for n in names:
            if n.startswith('CHANGELOG') or n.startswith('README.md') or '.test.' in n or ('.stories.' in n) or ('.spec.' in n):
                ignored.append(n)
        return ignored
    shutil.copytree(str(p / '_frontend_code' / component.js_dir), frontend, dirs_exist_ok=True, ignore=ignore)
    source_package_json = json.loads(Path(frontend / 'package.json').read_text())
    source_package_json['name'] = package_name
    source_package_json = _modify_js_deps(source_package_json, 'dependencies', p)
    source_package_json = _modify_js_deps(source_package_json, 'devDependencies', p)
    (frontend / 'package.json').write_text(json.dumps(source_package_json, indent=2))
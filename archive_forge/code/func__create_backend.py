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
def _create_backend(name: str, component: ComponentFiles, directory: Path, package_name: str):

    def find_template_in_list(template, list_to_search):
        for item in list_to_search:
            if template.lower() == item.lower():
                return item
        return None
    lists_to_search = [(gradio.components.__all__, 'components'), (gradio.layouts.__all__, 'layouts'), (gradio._simple_templates.__all__, '_simple_templates')]
    correct_cased_template = None
    module = None
    for list_, module_name in lists_to_search:
        correct_cased_template = find_template_in_list(component.template, list_)
        if correct_cased_template:
            module = module_name
            break
    if not correct_cased_template:
        raise ValueError(f'Cannot find {component.template} in gradio.components, gradio.layouts, or gradio._simple_templates. Please pass in a valid component name via the --template option. It must match the name of the python class.')
    if not module:
        raise ValueError('Module not found')
    readme_contents = textwrap.dedent('\n# {package_name}\nA Custom Gradio component.\n\n## Example usage\n\n```python\nimport gradio as gr\nfrom {package_name} import {name}\n```\n').format(package_name=package_name, name=name)
    (directory / 'README.md').write_text(readme_contents)
    backend = directory / 'backend' / package_name
    backend.mkdir(exist_ok=True, parents=True)
    gitignore = Path(__file__).parent / 'files' / 'gitignore'
    gitignore_contents = gitignore.read_text()
    gitignore_dest = directory / '.gitignore'
    gitignore_dest.write_text(gitignore_contents)
    pyproject = Path(__file__).parent / 'files' / 'pyproject_.toml'
    pyproject_contents = pyproject.read_text()
    pyproject_dest = directory / 'pyproject.toml'
    pyproject_contents = pyproject_contents.replace('<<name>>', package_name).replace('<<template>>', PATTERN.format(template=correct_cased_template))
    pyproject_dest.write_text(pyproject_contents)
    demo_dir = directory / 'demo'
    demo_dir.mkdir(exist_ok=True, parents=True)
    (demo_dir / 'app.py').write_text(f'\nimport gradio as gr\nfrom {package_name} import {name}\n\n{component.demo_code.format(name=name)}\n\nif __name__ == "__main__":\n    demo.launch()\n')
    (demo_dir / '__init__.py').touch()
    init = backend / '__init__.py'
    init.write_text(f"\nfrom .{name.lower()} import {name}\n\n__all__ = ['{name}']\n")
    p = Path(inspect.getfile(gradio)).parent
    python_file = backend / f'{name.lower()}.py'
    shutil.copy(str(p / module / component.python_file_name), str(python_file))
    source_pyi_file = p / module / component.python_file_name.replace('.py', '.pyi')
    pyi_file = backend / f'{name.lower()}.pyi'
    if source_pyi_file.exists():
        shutil.copy(str(source_pyi_file), str(pyi_file))
    content = python_file.read_text()
    content = _replace_old_class_name(correct_cased_template, name, content)
    content = _strip_document_lines(content)
    python_file.write_text(content)
    if pyi_file.exists():
        pyi_content = pyi_file.read_text()
        pyi_content = _replace_old_class_name(correct_cased_template, name, content)
        pyi_content = _strip_document_lines(pyi_content)
        pyi_file.write_text(pyi_content)
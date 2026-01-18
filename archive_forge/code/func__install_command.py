from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from rich.markup import escape
from typer import Argument, Option
from typing_extensions import Annotated
from gradio.cli.commands.display import LivePanelDisplay
from gradio.utils import set_directory
def _install_command(directory: Path, live: LivePanelDisplay, npm_install: str, pip_path: str | None):
    pip_executable_path = _get_executable_path('pip', executable_path=pip_path, cli_arg_name='--pip-path', check_3=True)
    cmds = [pip_executable_path, 'install', '-e', f'{str(directory)}[dev]']
    live.update(f':construction_worker: Installing python... [grey37]({escape(' '.join(cmds))})[/]')
    pipe = subprocess.run(cmds, capture_output=True, text=True, check=False)
    if pipe.returncode != 0:
        live.update(':red_square: Python installation [bold][red]failed[/][/]')
        live.update(pipe.stderr)
    else:
        live.update(':white_check_mark: Python install succeeded!')
    live.update(f':construction_worker: Installing javascript... [grey37]({npm_install})[/]')
    with set_directory(directory / 'frontend'):
        pipe = subprocess.run(npm_install.split(), capture_output=True, text=True, check=False)
        if pipe.returncode != 0:
            live.update(':red_square: NPM install [bold][red]failed[/][/]')
            live.update(pipe.stdout)
            live.update(pipe.stderr)
        else:
            live.update(':white_check_mark: NPM install succeeded!')
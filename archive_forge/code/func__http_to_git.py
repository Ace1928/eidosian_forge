import shutil
from pathlib import Path
from typing import Tuple
from wasabi import msg
from .commands import run_command
from .filesystem import is_subpath_of, make_tempdir
def _http_to_git(repo: str) -> str:
    if repo.startswith('http://'):
        repo = repo.replace('http://', 'https://')
    if repo.startswith('https://'):
        repo = repo.replace('https://', 'git@').replace('/', ':', 1)
        if repo.endswith('/'):
            repo = repo[:-1]
        repo = f'{repo}.git'
    return repo
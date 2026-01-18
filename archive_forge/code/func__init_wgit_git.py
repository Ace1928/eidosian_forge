from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple
import pygit2
def _init_wgit_git(self, gitignore: List) -> None:
    """
        Initializes a .git within .wgit directory, making it a git repo.

        Args:
            gitignore (List)
                a list of file paths to be ignored by the wgit git repo.
        """
    self.repo = pygit2.init_repository(str(self._parent_path), False)
    self.path = self._parent_path.joinpath('.git')
    self._parent_path.joinpath('.gitignore').touch(exist_ok=False)
    with open(self._parent_path.joinpath('.gitignore'), 'a') as file:
        for item in gitignore:
            file.write(f'{item}\n')
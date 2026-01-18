from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple
import pygit2
def _set_author_config(self, name: str, email: str) -> Tuple[str, str]:
    """Set the name and email for the pygit repo collecting from the gitconfig.
        If not available in gitconfig, set the values from the passed arguments."""
    gitconfig = Path('~/.gitconfig').expanduser()
    try:
        set_name = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.rstrip()
        set_email = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True).stdout.rstrip()
        if not set_name or not set_email:
            set_name = name
            set_email = email
    except BaseException:
        set_name = name
        set_email = email
    return (set_name, set_email)
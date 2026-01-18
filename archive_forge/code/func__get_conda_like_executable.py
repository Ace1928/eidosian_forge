import functools
import re
import shlex
import sys
from pathlib import Path
from IPython.core.magic import Magics, magics_class, line_magic
def _get_conda_like_executable(command):
    """Find the path to the given executable

    Parameters
    ----------

    executable: string
        Value should be: conda, mamba or micromamba
    """
    executable = Path(sys.executable).parent / command
    if executable.is_file():
        return str(executable)
    history = Path(sys.prefix, 'conda-meta', 'history').read_text(encoding='utf-8')
    match = re.search(f'^#\\s*cmd:\\s*(?P<command>.*{executable})\\s[create|install]', history, flags=re.MULTILINE)
    if match:
        return match.groupdict()['command']
    return command
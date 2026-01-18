import re
import subprocess
from typing import List, Optional
from ..constants import ENDPOINT
from ._subprocess import run_interactive_subprocess, run_subprocess
Parse the output of `git credential fill` to extract the password.

    Args:
        output (`str`):
            The output of `git credential fill`.
    
import re
import subprocess
from typing import List, Optional
from ..constants import ENDPOINT
from ._subprocess import run_interactive_subprocess, run_subprocess
def _parse_credential_output(output: str) -> List[str]:
    """Parse the output of `git credential fill` to extract the password.

    Args:
        output (`str`):
            The output of `git credential fill`.
    """
    return sorted(set((match[0] for match in GIT_CREDENTIAL_REGEX.findall(output))))
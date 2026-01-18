import os
import subprocess
from functools import partial
from getpass import getpass
from pathlib import Path
from typing import Optional
from . import constants
from .commands._cli_utils import ANSI
from .utils import (
from .utils._token import _get_token_from_environment, _get_token_from_google_colab
def _set_store_as_git_credential_helper_globally() -> None:
    """Set globally the credential.helper to `store`.

    To be used only in Google Colab as we assume the user doesn't care about the git
    credential config. It is the only particular case where we don't want to display the
    warning message in [`notebook_login()`].

    Related:
    - https://github.com/huggingface/huggingface_hub/issues/1043
    - https://github.com/huggingface/huggingface_hub/issues/1051
    - https://git-scm.com/docs/git-credential-store
    """
    try:
        run_subprocess('git config --global credential.helper store')
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
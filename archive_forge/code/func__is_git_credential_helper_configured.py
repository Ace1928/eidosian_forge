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
def _is_git_credential_helper_configured() -> bool:
    """Check if a git credential helper is configured.

    Warns user if not the case (except for Google Colab where "store" is set by default
    by `huggingface_hub`).
    """
    helpers = list_credential_helpers()
    if len(helpers) > 0:
        return True
    if is_google_colab():
        _set_store_as_git_credential_helper_globally()
        return True
    print(ANSI.red("Cannot authenticate through git-credential as no helper is defined on your machine.\nYou might have to re-authenticate when pushing to the Hugging Face Hub.\nRun the following command in your terminal in case you want to set the 'store' credential helper as default.\n\ngit config --global credential.helper store\n\nRead https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details."))
    return False
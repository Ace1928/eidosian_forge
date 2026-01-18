import os
import warnings
from pathlib import Path
from threading import Lock
from typing import Optional
from .. import constants
from ._runtime import is_google_colab
def _get_token_from_google_colab() -> Optional[str]:
    """Get token from Google Colab secrets vault using `google.colab.userdata.get(...)`.

    Token is read from the vault only once per session and then stored in a global variable to avoid re-requesting
    access to the vault.
    """
    if not is_google_colab():
        return None
    with _GOOGLE_COLAB_SECRET_LOCK:
        global _GOOGLE_COLAB_SECRET
        global _IS_GOOGLE_COLAB_CHECKED
        if _IS_GOOGLE_COLAB_CHECKED:
            return _GOOGLE_COLAB_SECRET
        try:
            from google.colab import userdata
            from google.colab.errors import Error as ColabError
        except ImportError:
            return None
        try:
            token = userdata.get('HF_TOKEN')
            _GOOGLE_COLAB_SECRET = _clean_token(token)
        except userdata.NotebookAccessError:
            warnings.warn('\nAccess to the secret `HF_TOKEN` has not been granted on this notebook.\nYou will not be requested again.\nPlease restart the session if you want to be prompted again.')
            _GOOGLE_COLAB_SECRET = None
        except userdata.SecretNotFoundError:
            warnings.warn('\nThe secret `HF_TOKEN` does not exist in your Colab secrets.\nTo authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\nYou will be able to reuse this secret in all of your notebooks.\nPlease note that authentication is recommended but still optional to access public models or datasets.')
            _GOOGLE_COLAB_SECRET = None
        except ColabError as e:
            warnings.warn(f"\nError while fetching `HF_TOKEN` secret value from your vault: '{str(e)}'.\nYou are not authenticated with the Hugging Face Hub in this notebook.\nIf the error persists, please let us know by opening an issue on GitHub (https://github.com/huggingface/huggingface_hub/issues/new).")
            _GOOGLE_COLAB_SECRET = None
        _IS_GOOGLE_COLAB_CHECKED = True
        return _GOOGLE_COLAB_SECRET
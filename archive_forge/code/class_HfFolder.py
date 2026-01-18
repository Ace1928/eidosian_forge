import warnings
from pathlib import Path
from typing import Optional
from .. import constants
from ._token import get_token
class HfFolder:
    path_token = Path(constants.HF_TOKEN_PATH)
    _old_path_token = Path(constants._OLD_HF_TOKEN_PATH)

    @classmethod
    def save_token(cls, token: str) -> None:
        """
        Save token, creating folder as needed.

        Token is saved in the huggingface home folder. You can configure it by setting
        the `HF_HOME` environment variable.

        Args:
            token (`str`):
                The token to save to the [`HfFolder`]
        """
        cls.path_token.parent.mkdir(parents=True, exist_ok=True)
        cls.path_token.write_text(token)

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.

        This method is deprecated in favor of [`huggingface_hub.get_token`] but is kept for backward compatibility.
        Its behavior is the same as [`huggingface_hub.get_token`].

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.
        """
        try:
            cls._copy_to_new_path_and_warn()
        except Exception:
            pass
        return get_token()

    @classmethod
    def delete_token(cls) -> None:
        """
        Deletes the token from storage. Does not fail if token does not exist.
        """
        try:
            cls.path_token.unlink()
        except FileNotFoundError:
            pass
        try:
            cls._old_path_token.unlink()
        except FileNotFoundError:
            pass

    @classmethod
    def _copy_to_new_path_and_warn(cls):
        if cls._old_path_token.exists() and (not cls.path_token.exists()):
            cls.save_token(cls._old_path_token.read_text())
            warnings.warn(f'A token has been found in `{cls._old_path_token}`. This is the old path where tokens were stored. The new location is `{cls.path_token}` which is configurable using `HF_HOME` environment variable. Your token has been copied to this new location. You can now safely delete the old token file manually or use `huggingface-cli logout`.')
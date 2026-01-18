from __future__ import annotations
import gettext as gettext_module
import os.path
from threading import local
def _pgettext(msgctxt: str, message: str) -> str:
    """Fetches a particular translation.

    It works with `msgctxt` .po modifiers and allows duplicate keys with different
    translations.

    Args:
        msgctxt (str): Context of the translation.
        message (str): Text to translate.

    Returns:
        str: Translated text.
    """
    return get_translation().pgettext(msgctxt, message)
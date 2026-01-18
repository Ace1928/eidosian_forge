import locale
from gettext import NullTranslations, translation
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
def _lazy_translate(catalog: str, namespace: str, message: str) -> str:
    """Used instead of _ when creating TranslationProxy, because _ is
    not bound yet at that time.
    """
    translator = get_translator(catalog, namespace)
    return translator.gettext(message)
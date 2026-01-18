from __future__ import annotations
import inspect
import re
import types
from typing import Any
def add_article(name: str, definite: bool=False, capital: bool=False) -> str:
    """Returns the string with a prepended article.

    The input does not need to begin with a character.

    Parameters
    ----------
    name : str
        Name to which to prepend an article
    definite : bool (default: False)
        Whether the article is definite or not.
        Indefinite articles being 'a' and 'an',
        while 'the' is definite.
    capital : bool (default: False)
        Whether the added article should have
        its first letter capitalized or not.
    """
    if definite:
        result = 'the ' + name
    else:
        first_letters = re.compile('[\\W_]+').sub('', name)
        if first_letters[:1].lower() in 'aeiou':
            result = 'an ' + name
        else:
            result = 'a ' + name
    if capital:
        return result[0].upper() + result[1:]
    else:
        return result
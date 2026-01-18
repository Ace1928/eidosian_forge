from ._base import *
@staticmethod
def _keyword_to_string(keywords: Filter) -> str:
    if type(keywords) == str:
        return f'"{keywords}"'
    else:
        return '(' + ' OR '.join((f'"{word}"' if ' ' in word else word for word in keywords)) + ')'
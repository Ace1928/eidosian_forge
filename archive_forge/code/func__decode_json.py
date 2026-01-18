from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
def _decode_json(cls, content, max_recursion_depth: int=100, recursion_depth: int=0):
    try:
        result = LazyJson.loads(content, recursive=True)
    except Exception as e:
        if recursion_depth >= max_recursion_depth:
            raise ValueError('Max Recursion depth is reached. JSON can´t be parsed!')
        idx_to_replace = int(e.pos)
        if isinstance(content, bytes):
            content.decode('utf-8')
        json_message = list(content)
        json_message[idx_to_replace] = ' '
        new_message = ''.join((str(m) for m in json_message))
        return GDELT._decode_json(content=new_message, max_recursion_depth=max_recursion_depth, recursion_depth=recursion_depth + 1)
    return result
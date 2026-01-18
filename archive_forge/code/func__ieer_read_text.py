import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def _ieer_read_text(s, root_label):
    stack = [Tree(root_label, [])]
    if s is None:
        return []
    for piece_m in re.finditer('<[^>]+>|[^\\s<]+', s):
        piece = piece_m.group()
        try:
            if piece.startswith('<b_'):
                m = _IEER_TYPE_RE.match(piece)
                if m is None:
                    print('XXXX', piece)
                chunk = Tree(m.group('type'), [])
                stack[-1].append(chunk)
                stack.append(chunk)
            elif piece.startswith('<e_'):
                stack.pop()
            else:
                stack[-1].append(piece)
        except (IndexError, ValueError) as e:
            raise ValueError(f'Bad IEER string (error at character {piece_m.start():d})') from e
    if len(stack) != 1:
        raise ValueError('Bad IEER string')
    return stack[0]
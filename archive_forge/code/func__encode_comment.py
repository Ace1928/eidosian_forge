import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _encode_comment(self, s=''):
    """(INTERNAL) Encodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        If ``s`` is None, this method will simply return an empty comment.

        :param s: (OPTIONAL) string.
        :return: a string with the encoded comment line.
        """
    if s:
        return '%s %s' % (_TK_COMMENT, s)
    else:
        return '%s' % _TK_COMMENT
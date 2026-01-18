import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _decode_relation(self, s):
    """(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string. The string must
        start with alphabetic character and must be quoted if the name includes
        spaces, otherwise this method will raise a `BadRelationFormat` exception.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r
" characters.

        :param s: a normalized string.
        :return: a string with the decoded relation name.
        """
    _, v = s.split(' ', 1)
    v = v.strip()
    if not _RE_RELATION.match(v):
        raise BadRelationFormat()
    res = str(v.strip('"\''))
    return res
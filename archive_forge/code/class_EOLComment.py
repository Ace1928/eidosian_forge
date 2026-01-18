from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
class EOLComment(CommentBase):
    name = 'EOLC'

    def __init__(self, value: Any, line: Any, column: Any) -> None:
        super().__init__(value, line, column)
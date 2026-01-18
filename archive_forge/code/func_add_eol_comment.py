from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def add_eol_comment(self, comment: Any, column: Any, line: Any) -> Any:
    if comment.count('\n') == 1:
        assert comment[-1] == '\n'
    else:
        assert '\n' not in comment
    self.comments[line] = retval = EOLComment(comment[:-1], line, column)
    self.unused.append(line)
    return retval
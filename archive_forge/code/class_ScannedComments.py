from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
class ScannedComments:

    def __init__(self: Any) -> None:
        self.comments = {}
        self.unused = []

    def add_eol_comment(self, comment: Any, column: Any, line: Any) -> Any:
        if comment.count('\n') == 1:
            assert comment[-1] == '\n'
        else:
            assert '\n' not in comment
        self.comments[line] = retval = EOLComment(comment[:-1], line, column)
        self.unused.append(line)
        return retval

    def add_blank_line(self, comment: Any, column: Any, line: Any) -> Any:
        assert comment.count('\n') == 1 and comment[-1] == '\n'
        assert line not in self.comments
        self.comments[line] = retval = BlankLineComment(comment[:-1], line, column)
        self.unused.append(line)
        return retval

    def add_full_line_comment(self, comment: Any, column: Any, line: Any) -> Any:
        assert comment.count('\n') == 1 and comment[-1] == '\n'
        self.comments[line] = retval = FullLineComment(comment[:-1], line, column)
        self.unused.append(line)
        return retval

    def __getitem__(self, idx: Any) -> Any:
        return self.comments[idx]

    def __str__(self) -> Any:
        return 'ParsedComments:\n  ' + '\n  '.join((f'{lineno:2} {x.info()}' for lineno, x in self.comments.items())) + '\n'

    def last(self) -> str:
        lineno, x = list(self.comments.items())[-1]
        return f'{lineno:2} {x.info()}\n'

    def any_unprocessed(self) -> bool:
        return len(self.unused) > 0

    def unprocessed(self, use: Any=False) -> Any:
        while len(self.unused) > 0:
            if _debug != 0:
                import inspect
                first = self.unused.pop(0) if use else self.unused[0]
                info = inspect.getframeinfo(inspect.stack()[1][0])
                xprintf('using', first, self.comments[first].value, info.function, info.lineno)
            yield (first, self.comments[first])
            if use:
                self.comments[first].set_used()

    def assign_pre(self, token: Any) -> Any:
        token_line = token.start_mark.line
        if _debug != 0:
            import inspect
            info = inspect.getframeinfo(inspect.stack()[1][0])
            xprintf('assign_pre', token_line, self.unused, info.function, info.lineno)
        gobbled = False
        while self.unused and self.unused[0] < token_line:
            gobbled = True
            first = self.unused.pop(0)
            if _debug != 0:
                xprintf('assign_pre < ', first)
            self.comments[first].set_used()
            token.add_comment_pre(first)
        return gobbled

    def assign_eol(self, tokens: Any) -> Any:
        try:
            comment_line = self.unused[0]
        except IndexError:
            return
        if not isinstance(self.comments[comment_line], EOLComment):
            return
        idx = 1
        while tokens[-idx].start_mark.line > comment_line or isinstance(tokens[-idx], ValueToken):
            idx += 1
        if _debug != 0:
            xprintf('idx1', idx)
        if len(tokens) > idx and isinstance(tokens[-idx], ScalarToken) and isinstance(tokens[-(idx + 1)], ScalarToken):
            return
        try:
            if isinstance(tokens[-idx], ScalarToken) and isinstance(tokens[-(idx + 1)], KeyToken):
                try:
                    eol_idx = self.unused.pop(0)
                    self.comments[eol_idx].set_used()
                    if _debug != 0:
                        xprintf('>>>>>a', idx, eol_idx, KEYCMNT)
                    tokens[-idx].add_comment_eol(eol_idx, KEYCMNT)
                except IndexError:
                    raise NotImplementedError
                return
        except IndexError:
            if _debug != 0:
                xprintf('IndexError1')
            pass
        try:
            if isinstance(tokens[-idx], ScalarToken) and isinstance(tokens[-(idx + 1)], (ValueToken, BlockEntryToken)):
                try:
                    eol_idx = self.unused.pop(0)
                    self.comments[eol_idx].set_used()
                    tokens[-idx].add_comment_eol(eol_idx, VALUECMNT)
                except IndexError:
                    raise NotImplementedError
                return
        except IndexError:
            if _debug != 0:
                xprintf('IndexError2')
            pass
        for t in tokens:
            xprintf('tt-', t)
        if _debug != 0:
            xprintf('not implemented EOL', type(tokens[-idx]))
        import sys
        sys.exit(0)

    def assign_post(self, token: Any) -> Any:
        token_line = token.start_mark.line
        if _debug != 0:
            import inspect
            info = inspect.getframeinfo(inspect.stack()[1][0])
            xprintf('assign_post', token_line, self.unused, info.function, info.lineno)
        gobbled = False
        while self.unused and self.unused[0] < token_line:
            gobbled = True
            first = self.unused.pop(0)
            if _debug != 0:
                xprintf('assign_post < ', first)
            self.comments[first].set_used()
            token.add_comment_post(first)
        return gobbled

    def str_unprocessed(self) -> Any:
        return ''.join((f'  {ind:2} {x.info()}\n' for ind, x in self.comments.items() if x.used == ' '))
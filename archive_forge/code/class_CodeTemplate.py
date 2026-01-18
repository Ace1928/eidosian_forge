import re
from typing import Mapping, Match, Optional, Sequence
class CodeTemplate:
    substitution_str = '(^[^\\n\\S]*)?\\$([^\\d\\W]\\w*|\\{,?[^\\d\\W]\\w*\\,?})'
    substitution = re.compile(substitution_str, re.MULTILINE)
    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> 'CodeTemplate':
        with open(filename) as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str='') -> None:
        self.pattern = pattern
        self.filename = filename

    def substitute(self, env: Optional[Mapping[str, object]]=None, **kwargs: object) -> str:
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            assert env is not None
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            return ''.join([indent + l + '\n' for e in v for l in str(e).splitlines()]).rstrip()

        def replace(match: Match[str]) -> str:
            indent = match.group(1)
            key = match.group(2)
            comma_before = ''
            comma_after = ''
            if key[0] == '{':
                key = key[1:-1]
                if key[0] == ',':
                    comma_before = ', '
                    key = key[1:]
                if key[-1] == ',':
                    comma_after = ', '
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ', '.join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)
        return self.substitution.sub(replace, self.pattern)
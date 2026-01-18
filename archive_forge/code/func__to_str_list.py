from typing import Any, List
def _to_str_list(self, indent_width: int=0) -> List[str]:
    codes: List[str] = []
    codes.append(' ' * indent_width + self._head + '{')
    for code in self._codes:
        next_indent_width = indent_width + 2
        if isinstance(code, str):
            codes.append(' ' * next_indent_width + code)
        elif isinstance(code, CodeBlock):
            codes += code._to_str_list(indent_width=next_indent_width)
        else:
            assert False
    codes.append(' ' * indent_width + '}')
    return codes
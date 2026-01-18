import base64
import mimetypes
import os
from html import escape
from typing import Any, Callable, Dict, Iterable, Match, Optional, Tuple
import bs4
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from nbconvert.filters.strings import add_anchor
class MathBlockParser(BlockParser):
    """This acts as a pass-through to the MathInlineParser. It is needed in
        order to avoid other block level rules splitting math sections apart.
        """
    MULTILINE_MATH = re.compile('(?<!\\\\)[$]{2}.*?(?<!\\\\)[$]{2}|\\\\\\\\\\[.*?\\\\\\\\\\]|\\\\begin\\{([a-z]*\\*?)\\}.*?\\\\end\\{\\1\\}', re.DOTALL)
    AXT_HEADING = re.compile(' {0,3}(#{1,6})(?!#+)(?: *\\n+|([^\\n]*?)(?:\\n+|\\s+?#+\\s*\\n+))')
    RULE_NAMES = ('multiline_math', *BlockParser.RULE_NAMES)

    def parse_multiline_math(self, m: Match[str], state: Any) -> Dict[str, str]:
        """Pass token through mutiline math."""
        return {'type': 'multiline_math', 'text': m.group(0)}
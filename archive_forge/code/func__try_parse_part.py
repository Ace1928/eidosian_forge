import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _try_parse_part(self, until_line):
    """
        Sets up a normal parser that uses a spezialized tokenizer to only parse
        until a certain position (or a bit longer if the statement hasn't
        ended.
        """
    self._parser_count += 1
    parsed_until_line = self._nodes_tree.parsed_until_line
    lines_after = self._parser_lines_new[parsed_until_line:]
    tokens = self._diff_tokenize(lines_after, until_line, line_offset=parsed_until_line)
    self._active_parser = Parser(self._pgen_grammar, error_recovery=True)
    return self._active_parser.parse(tokens=tokens)
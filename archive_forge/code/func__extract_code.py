import ast
from typing import Any, List, Tuple
from langchain_community.document_loaders.parsers.language.code_segmenter import (
def _extract_code(self, node: Any) -> str:
    start = node.lineno - 1
    end = node.end_lineno
    return '\n'.join(self.source_lines[start:end])
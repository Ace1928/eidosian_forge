import re
from typing import Callable, List
from langchain_community.document_loaders.parsers.language.code_segmenter import (
def _process_lines(self, func: Callable) -> List[str]:
    """A generic function to process COBOL lines based on provided func."""
    elements: List[str] = []
    start_idx = None
    inside_relevant_section = False
    for i, line in enumerate(self.source_lines):
        if self._is_relevant_code(line):
            inside_relevant_section = True
        if inside_relevant_section and (self.PARAGRAPH_PATTERN.match(line.strip().split(' ')[0]) or self.SECTION_PATTERN.match(line.strip())):
            if start_idx is not None:
                func(elements, start_idx, i)
            start_idx = i
    if start_idx is not None:
        func(elements, start_idx, len(self.source_lines))
    return elements
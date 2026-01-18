from __future__ import annotations
import re
from typing import Callable, Iterable, NamedTuple
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from .base import CompleteEvent, Completer, Completion
from .word_completer import WordCompleter
def _get_fuzzy_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
    word_before_cursor = document.get_word_before_cursor(pattern=re.compile(self._get_pattern()))
    document2 = Document(text=document.text[:document.cursor_position - len(word_before_cursor)], cursor_position=document.cursor_position - len(word_before_cursor))
    inner_completions = list(self.completer.get_completions(document2, complete_event))
    fuzzy_matches: list[_FuzzyMatch] = []
    if word_before_cursor == '':
        fuzzy_matches = [_FuzzyMatch(0, 0, compl) for compl in inner_completions]
    else:
        pat = '.*?'.join(map(re.escape, word_before_cursor))
        pat = f'(?=({pat}))'
        regex = re.compile(pat, re.IGNORECASE)
        for compl in inner_completions:
            matches = list(regex.finditer(compl.text))
            if matches:
                best = min(matches, key=lambda m: (m.start(), len(m.group(1))))
                fuzzy_matches.append(_FuzzyMatch(len(best.group(1)), best.start(), compl))

        def sort_key(fuzzy_match: _FuzzyMatch) -> tuple[int, int]:
            """Sort by start position, then by the length of the match."""
            return (fuzzy_match.start_pos, fuzzy_match.match_length)
        fuzzy_matches = sorted(fuzzy_matches, key=sort_key)
    for match in fuzzy_matches:
        yield Completion(text=match.completion.text, start_position=match.completion.start_position - len(word_before_cursor), display_meta=match.completion._display_meta, display=self._get_display(match, word_before_cursor), style=match.completion.style)
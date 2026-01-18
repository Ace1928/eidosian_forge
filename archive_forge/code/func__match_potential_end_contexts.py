import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _match_potential_end_contexts(self, text: str) -> Iterator[Tuple[Match, str]]:
    """
        Given a text, find the matches of potential sentence breaks,
        alongside the contexts surrounding these sentence breaks.

        Since the fix for the ReDOS discovered in issue #2866, we no longer match
        the word before a potential end of sentence token. Instead, we use a separate
        regex for this. As a consequence, `finditer`'s desire to find non-overlapping
        matches no longer aids us in finding the single longest match.
        Where previously, we could use::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._lang_vars.period_context_re().finditer(text)) # doctest: +SKIP
            [<re.Match object; span=(9, 18), match='acting!!!'>]

        Now we have to find the word before (i.e. 'acting') separately, and `finditer`
        returns::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._lang_vars.period_context_re().finditer(text)) # doctest: +NORMALIZE_WHITESPACE
            [<re.Match object; span=(15, 16), match='!'>,
            <re.Match object; span=(16, 17), match='!'>,
            <re.Match object; span=(17, 18), match='!'>]

        So, we need to find the word before the match from right to left, and then manually remove
        the overlaps. That is what this method does::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._match_potential_end_contexts(text))
            [(<re.Match object; span=(17, 18), match='!'>, 'acting!!! I')]

        :param text: String of one or more sentences
        :type text: str
        :return: Generator of match-context tuples.
        :rtype: Iterator[Tuple[Match, str]]
        """
    previous_slice = slice(0, 0)
    previous_match = None
    for match in self._lang_vars.period_context_re().finditer(text):
        before_text = text[previous_slice.stop:match.start()]
        index_after_last_space = self._get_last_whitespace_index(before_text)
        if index_after_last_space:
            index_after_last_space += previous_slice.stop + 1
        else:
            index_after_last_space = previous_slice.start
        prev_word_slice = slice(index_after_last_space, match.start())
        if previous_match and previous_slice.stop <= prev_word_slice.start:
            yield (previous_match, text[previous_slice] + previous_match.group() + previous_match.group('after_tok'))
        previous_match = match
        previous_slice = prev_word_slice
    if previous_match:
        yield (previous_match, text[previous_slice] + previous_match.group() + previous_match.group('after_tok'))
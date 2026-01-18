import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
class PunktBaseClass:
    """
    Includes common components of PunktTrainer and PunktSentenceTokenizer.
    """

    def __init__(self, lang_vars=None, token_cls=PunktToken, params=None):
        if lang_vars is None:
            lang_vars = PunktLanguageVars()
        if params is None:
            params = PunktParameters()
        self._params = params
        self._lang_vars = lang_vars
        self._Token = token_cls
        'The collection of parameters that determines the behavior\n        of the punkt tokenizer.'

    def _tokenize_words(self, plaintext):
        """
        Divide the given text into tokens, using the punkt word
        segmentation regular expression, and generate the resulting list
        of tokens augmented as three-tuples with two boolean values for whether
        the given token occurs at the start of a paragraph or a new line,
        respectively.
        """
        parastart = False
        for line in plaintext.split('\n'):
            if line.strip():
                line_toks = iter(self._lang_vars.word_tokenize(line))
                try:
                    tok = next(line_toks)
                except StopIteration:
                    continue
                yield self._Token(tok, parastart=parastart, linestart=True)
                parastart = False
                for tok in line_toks:
                    yield self._Token(tok)
            else:
                parastart = True

    def _annotate_first_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform the first pass of annotation, which makes decisions
        based purely based on the word type of each word:

          - '?', '!', and '.' are marked as sentence breaks.
          - sequences of two or more periods are marked as ellipsis.
          - any word ending in '.' that's a known abbreviation is
            marked as an abbreviation.
          - any other word ending in '.' is marked as a sentence break.

        Return these annotations as a tuple of three sets:

          - sentbreak_toks: The indices of all sentence breaks.
          - abbrev_toks: The indices of all abbreviations.
          - ellipsis_toks: The indices of all ellipsis marks.
        """
        for aug_tok in tokens:
            self._first_pass_annotation(aug_tok)
            yield aug_tok

    def _first_pass_annotation(self, aug_tok: PunktToken) -> None:
        """
        Performs type-based annotation on a single token.
        """
        tok = aug_tok.tok
        if tok in self._lang_vars.sent_end_chars:
            aug_tok.sentbreak = True
        elif aug_tok.is_ellipsis:
            aug_tok.ellipsis = True
        elif aug_tok.period_final and (not tok.endswith('..')):
            if tok[:-1].lower() in self._params.abbrev_types or tok[:-1].lower().split('-')[-1] in self._params.abbrev_types:
                aug_tok.abbr = True
            else:
                aug_tok.sentbreak = True
        return
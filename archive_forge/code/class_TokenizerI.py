from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple
from nltk.internals import overridden
from nltk.tokenize.util import string_span_tokenize
class TokenizerI(ABC):
    """
    A processing interface for tokenizing a string.
    Subclasses must define ``tokenize()`` or ``tokenize_sents()`` (or both).
    """

    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of *s*.

        :rtype: List[str]
        """
        if overridden(self.tokenize_sents):
            return self.tokenize_sents([s])[0]

    def span_tokenize(self, s: str) -> Iterator[Tuple[int, int]]:
        """
        Identify the tokens using integer offsets ``(start_i, end_i)``,
        where ``s[start_i:end_i]`` is the corresponding token.

        :rtype: Iterator[Tuple[int, int]]
        """
        raise NotImplementedError()

    def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
        """
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

            return [self.tokenize(s) for s in strings]

        :rtype: List[List[str]]
        """
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(self, strings: List[str]) -> Iterator[List[Tuple[int, int]]]:
        """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

            return [self.span_tokenize(s) for s in strings]

        :yield: List[Tuple[int, int]]
        """
        for s in strings:
            yield list(self.span_tokenize(s))
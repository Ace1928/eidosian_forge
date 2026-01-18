from typing import NamedTuple, Dict, Union, Iterator, Any
from emoji import unicode_codes
class EmojiMatchZWJ(EmojiMatch):
    """
    Represents a match of multiple emoji in a string that were joined by
    zero-width-joiners (ZWJ/``\\u200D``)."""
    __slots__ = ('emojis',)

    def __init__(self, match: EmojiMatch):
        super().__init__(match.emoji, match.start, match.end, match.data)
        self.emojis = []
        'List of sub emoji as EmojiMatch objects'
        i = match.start
        for e in match.emoji.split(_ZWJ):
            m = EmojiMatch(e, i, i + len(e), unicode_codes.EMOJI_DATA.get(e, None))
            self.emojis.append(m)
            i += len(e) + 1

    def join(self) -> str:
        """
        Joins a ZWJ-emoji into a string
        """
        return _ZWJ.join((e.emoji for e in self.emojis))

    def is_zwj(self) -> bool:
        return True

    def split(self) -> 'EmojiMatchZWJ':
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.join()}, {self.start}:{self.end})'
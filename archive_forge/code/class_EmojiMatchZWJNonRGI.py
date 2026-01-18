from typing import NamedTuple, Dict, Union, Iterator, Any
from emoji import unicode_codes
class EmojiMatchZWJNonRGI(EmojiMatchZWJ):
    """
    Represents a match of multiple emoji in a string that were joined by
    zero-width-joiners (ZWJ/``\\u200D``). This class is only used for emoji
    that are not "recommended for general interchange" (non-RGI) by Unicode.org.
    The data property of this class is always None.
    """

    def __init__(self, first_emoji_match: EmojiMatch, second_emoji_match: EmojiMatch):
        self.emojis = [first_emoji_match, second_emoji_match]
        'List of sub emoji as EmojiMatch objects'
        self._update()

    def _update(self):
        self.emoji = _ZWJ.join((e.emoji for e in self.emojis))
        self.start = self.emojis[0].start
        self.end = self.emojis[-1].end
        self.data = None

    def _add(self, next_emoji_match: EmojiMatch):
        self.emojis.append(next_emoji_match)
        self._update()
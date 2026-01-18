import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class FunctionalPseudoElement:
    """
    Represents selector::name(arguments)

    .. attribute:: name

        The name (identifier) of the pseudo-element, as a string.

    .. attribute:: arguments

        The arguments of the pseudo-element, as a list of tokens.

        **Note:** tokens are not part of the public API,
        and may change between cssselect versions.
        Use at your own risks.

    """

    def __init__(self, name: str, arguments: Sequence['Token']):
        self.name = ascii_lower(name)
        self.arguments = arguments

    def __repr__(self) -> str:
        return '%s[::%s(%r)]' % (self.__class__.__name__, self.name, [token.value for token in self.arguments])

    def argument_types(self) -> List[str]:
        return [token.type for token in self.arguments]

    def canonical(self) -> str:
        args = ''.join((token.css() for token in self.arguments))
        return '%s(%s)' % (self.name, args)
import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class EOFToken(Token):

    def __new__(cls, pos: int) -> 'EOFToken':
        return typing.cast('EOFToken', Token.__new__(cls, 'EOF', None, pos))

    def __repr__(self) -> str:
        return '<%s at %i>' % (self.type, self.pos)
import collections
import pyrfc3339
from ._conditions import (
 Parses a caveat into an identifier, identifying the checker that should
    be used, and the argument to the checker (the rest of the string).

    The identifier is taken from all the characters before the first
    space character.
    :return two string, identifier and arg
    
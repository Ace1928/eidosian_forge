from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _listToPhrase(things, finalDelimiter, delimiter=', '):
    """
    Produce a string containing each thing in C{things},
    separated by a C{delimiter}, with the last couple being separated
    by C{finalDelimiter}

    @param things: The elements of the resulting phrase
    @type things: L{list} or L{tuple}

    @param finalDelimiter: What to put between the last two things
        (typically 'and' or 'or')
    @type finalDelimiter: L{str}

    @param delimiter: The separator to use between each thing,
        not including the last two. Should typically include a trailing space.
    @type delimiter: L{str}

    @return: The resulting phrase
    @rtype: L{str}
    """
    if not isinstance(things, (list, tuple)):
        raise TypeError('Things must be a list or a tuple')
    if not things:
        return ''
    if len(things) == 1:
        return str(things[0])
    if len(things) == 2:
        return f'{str(things[0])} {finalDelimiter} {str(things[1])}'
    else:
        strThings = []
        for thing in things:
            strThings.append(str(thing))
        return '{}{}{} {}'.format(delimiter.join(strThings[:-1]), delimiter, finalDelimiter, strThings[-1])
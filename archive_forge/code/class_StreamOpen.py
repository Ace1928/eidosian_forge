from itertools import count
from typing import Dict, Iterator, List, TypeVar
from attrs import Factory, define
from twisted.protocols.amp import AMP, Command, Integer, String as Bytes
class StreamOpen(Command):
    """
    Open a new stream.
    """
    response = [(b'streamId', Integer())]
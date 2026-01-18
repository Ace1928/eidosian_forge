from itertools import count
from typing import Dict, Iterator, List, TypeVar
from attrs import Factory, define
from twisted.protocols.amp import AMP, Command, Integer, String as Bytes
class StreamWrite(Command):
    """
    Write a chunk of data to a stream.
    """
    arguments = [(b'streamId', Integer()), (b'data', Bytes())]
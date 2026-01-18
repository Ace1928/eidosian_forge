from abc import ABC, abstractmethod
from typing import List, Iterator
from .exceptions import LarkError
from .lark import PostLex
from .lexer import Token
@property
@abstractmethod
def INDENT_type(self) -> str:
    raise NotImplementedError()
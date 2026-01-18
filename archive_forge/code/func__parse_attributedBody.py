from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _parse_attributedBody(self, attributedBody: bytes) -> str:
    """
        Parse the attributedBody field of the message table
        for the text content of the message.

        The attributedBody field is a binary blob that contains
        the message content after the byte string b"NSString":

                              5 bytes      1-3 bytes    `len` bytes
        ... | b"NSString" |   preamble   |   `len`   |    contents    | ...

        The 5 preamble bytes are always b"\x01\x94\x84\x01+"

        The size of `len` is either 1 byte or 3 bytes:
        - If the first byte in `len` is b"\x81" then `len` is 3 bytes long.
          So the message length is the 2 bytes after, in little Endian.
        - Otherwise, the size of `len` is 1 byte, and the message length is
          that byte.

        Args:
            attributedBody (bytes): attributedBody field of the message table.
        Return:
            str: Text content of the message.
        """
    content = attributedBody.split(b'NSString')[1][5:]
    length, start = (content[0], 1)
    if content[0] == 129:
        length, start = (int.from_bytes(content[1:3], 'little'), 3)
    return content[start:start + length].decode('utf-8', errors='ignore')
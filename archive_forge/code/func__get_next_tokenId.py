import os
import re
import time
from enum import Enum
from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_next_tokenId(self, tokenId: str) -> str:
    value_type = self._detect_value_type(tokenId)
    if value_type == 'hex_0x':
        value_int = int(tokenId, 16)
    elif value_type == 'hex_0xbf':
        value_int = int(tokenId[2:], 16)
    else:
        value_int = int(tokenId)
    result = value_int + 1
    if value_type == 'hex_0x':
        return '0x' + format(result, '0' + str(len(tokenId) - 2) + 'x')
    elif value_type == 'hex_0xbf':
        return '0xbf' + format(result, '0' + str(len(tokenId) - 4) + 'x')
    else:
        return str(result)
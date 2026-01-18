from pathlib import Path
import urllib.parse
import os
from typing import Union
@classmethod
def _get_str_representation(cls, parsed_uri: urllib.parse.ParseResult, path: Union[str, Path]) -> str:
    if not parsed_uri.scheme:
        return str(path)
    return parsed_uri._replace(netloc=str(path), path='').geturl()
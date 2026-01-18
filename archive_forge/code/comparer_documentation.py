import difflib
import pathlib
from typing import Any, List, Union
compare self and other.

        When different is not exist, return empty list.

        >>> PathComparer('/to/index').diff('C:\to\index')
        []

        When different is exist, return unified diff style list as:

        >>> PathComparer('/to/index').diff('C:\to\index2')
        [
           '- C:/to/index'
           '+ C:/to/index2'
           '?            +'
        ]
        
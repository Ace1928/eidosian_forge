from unittest.mock import patch
from docstring_parser import parse_from_object
class StandardCase:
    """Short description
        Long description
        """
    attr_one: str
    'Description for attr_one'
    attr_two: bool = False
    'Description for attr_two'
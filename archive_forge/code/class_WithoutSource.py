from unittest.mock import patch
from docstring_parser import parse_from_object
class WithoutSource:
    """Short description"""
    attr_one: str
    'Description for attr_one'
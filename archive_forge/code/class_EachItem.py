import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
class EachItem(railroad.Group):
    """
    Custom railroad item to compose a:
    - Group containing a
      - OneOrMore containing a
        - Choice of the elements in the Each
    with the group label indicating that all must be matched
    """
    all_label = '[ALL]'

    def __init__(self, *items):
        choice_item = railroad.Choice(len(items) - 1, *items)
        one_or_more_item = railroad.OneOrMore(item=choice_item)
        super().__init__(one_or_more_item, label=self.all_label)
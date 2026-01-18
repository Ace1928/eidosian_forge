import markdown
from functools import reduce
from markdown.treeprocessors import Treeprocessor
import xml.etree.ElementTree as etree
class TaskListExtension(markdown.Extension):
    """
    An extension that supports GitHub task lists. Both ordered and unordered
    lists are supported and can be separately enabled. Nested lists are
    supported.

    Example::

       - [x] milk
       - [ ] eggs
       - [x] chocolate
       - [ ] if possible:
           1. [ ] solve world peace
           2. [ ] solve world hunger
    """

    def __init__(self, **kwargs):
        self.config = {'ordered': [True, 'Enable parsing of ordered lists'], 'unordered': [True, 'Enable parsing of unordered lists'], 'checked': ['[x]', 'The checked state pattern'], 'unchecked': ['[ ]', 'The unchecked state pattern'], 'max_depth': [0, 'Maximum list nesting depth (None for unlimited)'], 'list_attrs': [{}, 'Additional attribute dict (or callable) to add to the <ul> or <ol> element'], 'item_attrs': [{}, 'Additional attribute dict (or callable) to add to the <li> element'], 'checkbox_attrs': [{}, 'Additional attribute dict (or callable) to add to the checkbox <input> element']}
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        md.treeprocessors.register(TaskListProcessor(self), 'gfm-tasklist', 100)
from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _defined_facts_lines(self):
    yield 'defined_facts = ['
    for fact in sorted(self.defined_facts):
        yield f'    {fact!r},'
    yield '] # defined_facts'
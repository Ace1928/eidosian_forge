from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _full_implications_lines(self):
    yield 'full_implications = dict( ['
    for fact in sorted(self.defined_facts):
        for value in (True, False):
            yield f'    # Implications of {fact} = {value}:'
            yield f'    (({fact!r}, {value!r}), set( ('
            implications = self.full_implications[fact, value]
            for implied in sorted(implications):
                yield f'        {implied!r},'
            yield '       ) ),'
            yield '     ),'
    yield ' ] ) # full_implications'
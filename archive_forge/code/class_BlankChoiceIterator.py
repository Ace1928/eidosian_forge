from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import islice, tee, zip_longest
from django.utils.functional import Promise
class BlankChoiceIterator(BaseChoiceIterator):
    """Iterator to lazily inject a blank choice."""

    def __init__(self, choices, blank_choice):
        self.choices = choices
        self.blank_choice = blank_choice

    def __iter__(self):
        choices, other = tee(self.choices)
        if not any((value in ('', None) for value, _ in flatten_choices(other))):
            yield from self.blank_choice
        yield from choices
from __future__ import (absolute_import, division, print_function)
import copy
import functools
import itertools
import random
import sys
import time
import ansible.module_utils.compat.typing as t
def _emit_isolated_iterator_copies(original_iterator):
    _copiable_iterator, _first_iterator_copy = itertools.tee(original_iterator)
    yield _first_iterator_copy
    while True:
        yield copy.copy(_copiable_iterator)
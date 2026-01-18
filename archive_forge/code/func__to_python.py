from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _to_python(self) -> str:
    """ Generate a string with plain python representation of the instance """
    return '\n'.join(self.print_rules())
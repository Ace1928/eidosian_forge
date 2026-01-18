from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class TaggyTag:
    """Tag with a custom repr function to test circuit diagrams."""

    def __repr__(self):
        return 'TaggyTag()'
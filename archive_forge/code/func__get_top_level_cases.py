from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def _get_top_level_cases(self):
    test_cases = []
    test_cases.append(('basic', '{val}'))
    test_cases.append(('if', dedent('\n        if True:\n            {val}\n        ')))
    test_cases.append(('while', dedent('\n        while True:\n            {val}\n            break\n        ')))
    test_cases.append(('try', dedent('\n        try:\n            {val}\n        except:\n            pass\n        ')))
    test_cases.append(('except', dedent('\n        try:\n            pass\n        except:\n            {val}\n        ')))
    test_cases.append(('finally', dedent('\n        try:\n            pass\n        except:\n            pass\n        finally:\n            {val}\n        ')))
    test_cases.append(('for', dedent('\n        for _ in range(4):\n            {val}\n        ')))
    test_cases.append(('nested', dedent('\n        if True:\n            while True:\n                {val}\n                break\n        ')))
    test_cases.append(('deep-nested', dedent('\n        if True:\n            while True:\n                break\n                for x in range(3):\n                    if True:\n                        while True:\n                            for x in range(3):\n                                {val}\n        ')))
    return test_cases
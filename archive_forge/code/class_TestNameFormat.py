import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
@unique
class TestNameFormat(Enum):
    """
    An enum to configure how ``mk_test_name()`` to compose a test name.  Given
    the following example:

    .. code-block:: python

        @data("a", "b")
        def testSomething(self, value):
            ...

    if using just ``@ddt`` or together with ``DEFAULT``:

    * ``testSomething_1_a``
    * ``testSomething_2_b``

    if using ``INDEX_ONLY``:

    * ``testSomething_1``
    * ``testSomething_2``

    """
    DEFAULT = 0
    INDEX_ONLY = 1
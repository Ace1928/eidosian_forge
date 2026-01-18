from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
class TestStringItem(MemoryLeakMixin, TestCase):
    """Test list can take strings as items. """

    def test_string_item(self):

        @njit
        def foo():
            l = listobject.new_list(types.unicode_type)
            l.append('a')
            l.append('b')
            l.append('c')
            l.append('d')
            return (l[0], l[1], l[2], l[3])
        items = foo()
        self.assertEqual(['a', 'b', 'c', 'd'], list(items))
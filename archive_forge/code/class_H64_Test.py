from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
class H64_Test(_Base64Test):
    """test H64 codec functions"""
    engine = h64
    descriptionPrefix = 'h64 codec'
    encoded_data = [(b'', b''), (b'U', b'J/'), (b'U\xaa', b'Jd8'), (b'U\xaaU', b'JdOJ'), (b'U\xaaU\xaa', b'JdOJe0'), (b'U\xaaU\xaaU', b'JdOJeK3'), (b'U\xaaU\xaaU\xaa', b'JdOJeKZe'), (b'U\xaaU\xaf', b'JdOJj0'), (b'U\xaaU\xaa_', b'JdOJey3')]
    encoded_ints = [(b'z.', 63, 12), (b'.z', 4032, 12)]
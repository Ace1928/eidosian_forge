from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class _sha1_crypt_test(HandlerCase):
    handler = hash.sha1_crypt
    known_correct_hashes = [('password', '$sha1$19703$iVdJqfSE$v4qYKl1zqYThwpjJAoKX6UvlHq/a'), ('password', '$sha1$21773$uV7PTeux$I9oHnvwPZHMO0Nq6/WgyGV/tDJIH'), (UPASS_TABLE, '$sha1$40000$uJ3Sp7LE$.VEmLO5xntyRFYihC7ggd3297T/D')]
    known_malformed_hashes = ['$sha1$21773$u!7PTeux$I9oHnvwPZHMO0Nq6/WgyGV/tDJIH', '$sha1$01773$uV7PTeux$I9oHnvwPZHMO0Nq6/WgyGV/tDJIH', '$sha1$21773$uV7PTeux$I9oHnvwPZHMO0Nq6/WgyGV/tDJIH$', '$sha1$$uV7PTeux$I9oHnvwPZHMO0Nq6/WgyGV/tDJIH$']
    platform_crypt_support = [('netbsd', True), ('freebsd|openbsd|solaris|darwin', False), ('linux', None)]
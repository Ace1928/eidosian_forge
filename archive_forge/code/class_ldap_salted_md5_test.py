from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class ldap_salted_md5_test(HandlerCase):
    handler = hash.ldap_salted_md5
    known_correct_hashes = [('testing1234', '{SMD5}UjFY34os/pnZQ3oQOzjqGu4yeXE='), (UPASS_TABLE, '{SMD5}Z0ioJ58LlzUeRxm3K6JPGAvBGIM='), ('test', '{SMD5}LnuZPJhiaY95/4lmVFpg548xBsD4P4cw'), ('test', '{SMD5}XRlncfRzvGi0FDzgR98tUgBg7B3jXOs9p9S615qTkg=='), ('test', '{SMD5}FbAkzOMOxRbMp6Nn4hnZuel9j9Gas7a2lvI+x5hT6j0=')]
    known_malformed_hashes = ['{SMD5}IGVhwK+anvspmfDt2t0vgGjt/Q==', '{SMD5}LnuZPJhiaY95/4lmVFpg548xBsD4P4c', '{SMD5}LnuZPJhiaY95/4lmVFpg548xBsD4P4cw{SMD5}LnuZPJhiaY95/4lmVFpg548xBsD4P4cw=', '{SMD5}LnuZPJhiaY95/4lmV=pg548xBsD4P4cw', '{SMD5}LnuZPJhiaY95/4lmVFpg548xBsD4P===']
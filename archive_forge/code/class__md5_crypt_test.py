from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class _md5_crypt_test(HandlerCase):
    handler = hash.md5_crypt
    known_correct_hashes = [('U*U*U*U*', '$1$dXc3I7Rw$ctlgjDdWJLMT.qwHsWhXR1'), ('U*U***U', '$1$dXc3I7Rw$94JPyQc/eAgQ3MFMCoMF.0'), ('U*U***U*', '$1$dXc3I7Rw$is1mVIAEtAhIzSdfn5JOO0'), ('*U*U*U*U', '$1$eQT9Hwbt$XtuElNJD.eW5MN5UCWyTQ0'), ('', '$1$Eu.GHtia$CFkL/nE1BYTlEPiVx1VWX0'), ('', '$1$dOHYPKoP$tnxS1T8Q6VVn3kpV8cN6o.'), (' ', '$1$m/5ee7ol$bZn0kIBFipq39e.KDXX8I0'), ('test', '$1$ec6XvcoW$ghEtNK2U1MC5l.Dwgi3020'), ('Compl3X AlphaNu3meric', '$1$nX1e7EeI$ljQn72ZUgt6Wxd9hfvHdV0'), ('4lpHa N|_|M3r1K W/ Cur5Es: #$%(*)(*%#', '$1$jQS7o98J$V6iTcr71CGgwW2laf17pi1'), ('test', '$1$SuMrG47N$ymvzYjr7QcEQjaK5m1PGx1'), (b'test', '$1$SuMrG47N$ymvzYjr7QcEQjaK5m1PGx1'), (u('s'), '$1$ssssssss$YgmLTApYTv12qgTwBoj8i/'), (UPASS_TABLE, '$1$d6/Ky1lU$/xpf8m7ftmWLF.TjHCqel0')]
    known_malformed_hashes = ['$1$dOHYPKoP$tnxS1T8Q6VVn3kpV8cN6o!', '$1$dOHYPKoP$tnxS1T8Q6VVn3kpV8cN6o.$']
    platform_crypt_support = [('openbsd[6789]', False), ('openbsd5', None), ('openbsd', True), ('freebsd|netbsd|linux|solaris', True), ('darwin', False)]
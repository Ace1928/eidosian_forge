from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class msdcc_test(UserHandlerMixin, HandlerCase):
    handler = hash.msdcc
    user_case_insensitive = True
    known_correct_hashes = [(('Asdf999', 'sevans'), 'b1176c2587478785ec1037e5abc916d0'), (('ASDqwe123', 'jdoe'), '592cdfbc3f1ef77ae95c75f851e37166'), (('test1', 'test1'), '64cd29e36a8431a2b111378564a10631'), (('test2', 'test2'), 'ab60bdb4493822b175486810ac2abe63'), (('test3', 'test3'), '14dd041848e12fc48c0aa7a416a4a00c'), (('test4', 'test4'), 'b945d24866af4b01a6d89b9d932a153c'), (('1234qwer!@#$', 'Administrator'), '7b69d06ef494621e3f47b9802fe7776d'), (('password', 'user'), '2d9f0b052932ad18b87f315641921cda'), (('', 'root'), '176a4c2bd45ac73687676c2f09045353'), (('test1', 'TEST1'), '64cd29e36a8431a2b111378564a10631'), (('okolada', 'nineteen_characters'), '290efa10307e36a79b3eebf2a6b29455'), ((u('ü'), u('ü')), '48f84e6f73d6d5305f6558a33fa2c9bb'), ((u('üü'), u('üü')), '593246a8335cf0261799bda2a2a9c623'), ((u('€€'), 'user'), '9121790702dda0fa5d353014c334c2ce'), ((UPASS_TABLE, 'bob'), 'fcb82eb4212865c7ac3503156ca3f349')]
    known_alternate_hashes = [('B1176C2587478785EC1037E5ABC916D0', ('Asdf999', 'sevans'), 'b1176c2587478785ec1037e5abc916d0')]
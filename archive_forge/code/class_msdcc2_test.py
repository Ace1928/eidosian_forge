from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class msdcc2_test(UserHandlerMixin, HandlerCase):
    handler = hash.msdcc2
    user_case_insensitive = True
    known_correct_hashes = [(('test1', 'test1'), '607bbe89611e37446e736f7856515bf8'), (('qerwt', 'Joe'), 'e09b38f84ab0be586b730baf61781e30'), (('12345', 'Joe'), '6432f517a900b3fc34ffe57f0f346e16'), (('', 'bin'), 'c0cbe0313a861062e29f92ede58f9b36'), (('w00t', 'nineteen_characters'), '87136ae0a18b2dafe4a41d555425b2ed'), (('w00t', 'eighteencharacters'), 'fc5df74eca97afd7cd5abb0032496223'), (('longpassword', 'twentyXXX_characters'), 'cfc6a1e33eb36c3d4f84e4c2606623d2'), (('longpassword', 'twentyoneX_characters'), '99ff74cea552799da8769d30b2684bee'), (('longpassword', 'twentytwoXX_characters'), '0a721bdc92f27d7fb23b87a445ec562f'), (('test2', 'TEST2'), 'c6758e5be7fc943d00b97972a8a97620'), (('test3', 'test3'), '360e51304a2d383ea33467ab0b639cc4'), (('test4', 'test4'), '6f79ee93518306f071c47185998566ae'), ((u('ü'), 'joe'), 'bdb80f2c4656a8b8591bd27d39064a54'), ((u('€€'), 'joe'), '1e1e20f482ff748038e47d801d0d1bda'), ((u('üü'), 'admin'), '0839e4a07c00f18a8c65cf5b985b9e73'), ((UPASS_TABLE, 'bob'), 'cad511dc9edefcf69201da72efb6bb55')]
import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class TagSerializeTests(TestCase):

    def test_serialize_simple(self):
        x = make_object(Tag, tagger=b'Jelmer Vernooij <jelmer@samba.org>', name=b'0.1', message=b'Tag 0.1', object=(Blob, b'd80c186a03f423a81b39df39dc87fd269736ca86'), tag_time=423423423, tag_timezone=0)
        self.assertEqual(b'object d80c186a03f423a81b39df39dc87fd269736ca86\ntype blob\ntag 0.1\ntagger Jelmer Vernooij <jelmer@samba.org> 423423423 +0000\n\nTag 0.1', x.as_raw_string())

    def test_serialize_none_message(self):
        x = make_object(Tag, tagger=b'Jelmer Vernooij <jelmer@samba.org>', name=b'0.1', message=None, object=(Blob, b'd80c186a03f423a81b39df39dc87fd269736ca86'), tag_time=423423423, tag_timezone=0)
        self.assertEqual(b'object d80c186a03f423a81b39df39dc87fd269736ca86\ntype blob\ntag 0.1\ntagger Jelmer Vernooij <jelmer@samba.org> 423423423 +0000\n', x.as_raw_string())
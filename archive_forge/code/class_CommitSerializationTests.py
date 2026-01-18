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
class CommitSerializationTests(TestCase):

    def make_commit(self, **kwargs):
        attrs = {'tree': b'd80c186a03f423a81b39df39dc87fd269736ca86', 'parents': [b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'], 'author': b'James Westby <jw+debian@jameswestby.net>', 'committer': b'James Westby <jw+debian@jameswestby.net>', 'commit_time': 1174773719, 'author_time': 1174773719, 'commit_timezone': 0, 'author_timezone': 0, 'message': b'Merge ../b\n'}
        attrs.update(kwargs)
        return make_commit(**attrs)

    def test_encoding(self):
        c = self.make_commit(encoding=b'iso8859-1')
        self.assertIn(b'encoding iso8859-1\n', c.as_raw_string())

    def test_short_timestamp(self):
        c = self.make_commit(commit_time=30)
        c1 = Commit()
        c1.set_raw_string(c.as_raw_string())
        self.assertEqual(30, c1.commit_time)

    def test_full_tree(self):
        c = self.make_commit(commit_time=30)
        t = Tree()
        t.add(b'data-x', 420, Blob().id)
        c.tree = t
        c1 = Commit()
        c1.set_raw_string(c.as_raw_string())
        self.assertEqual(t.id, c1.tree)
        self.assertEqual(c.as_raw_string(), c1.as_raw_string())

    def test_raw_length(self):
        c = self.make_commit()
        self.assertEqual(len(c.as_raw_string()), c.raw_length())

    def test_simple(self):
        c = self.make_commit()
        self.assertEqual(c.id, b'5dac377bdded4c9aeb8dff595f0faeebcc8498cc')
        self.assertEqual(b'tree d80c186a03f423a81b39df39dc87fd269736ca86\nparent ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd\nparent 4cffe90e0a41ad3f5190079d7c8f036bde29cbe6\nauthor James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ncommitter James Westby <jw+debian@jameswestby.net> 1174773719 +0000\n\nMerge ../b\n', c.as_raw_string())

    def test_timezone(self):
        c = self.make_commit(commit_timezone=5 * 60)
        self.assertIn(b' +0005\n', c.as_raw_string())

    def test_neg_timezone(self):
        c = self.make_commit(commit_timezone=-1 * 3600)
        self.assertIn(b' -0100\n', c.as_raw_string())

    def test_deserialize(self):
        c = self.make_commit()
        d = Commit()
        d._deserialize(c.as_raw_chunks())
        self.assertEqual(c, d)

    def test_serialize_gpgsig(self):
        commit = self.make_commit(gpgsig=b'-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1\n\niQIcBAABCgAGBQJULCdfAAoJEACAbyvXKaRXuKwP/RyP9PA49uAvu8tQVCC/uBa8\nvi975+xvO14R8Pp8k2nps7lSxCdtCd+xVT1VRHs0wNhOZo2YCVoU1HATkPejqSeV\nNScTHcxnk4/+bxyfk14xvJkNp7FlQ3npmBkA+lbV0Ubr33rvtIE5jiJPyz+SgWAg\nxdBG2TojV0squj00GoH/euK6aX7GgZtwdtpTv44haCQdSuPGDcI4TORqR6YSqvy3\nGPE+3ZqXPFFb+KILtimkxitdwB7CpwmNse2vE3rONSwTvi8nq3ZoQYNY73CQGkUy\nqoFU0pDtw87U3niFin1ZccDgH0bB6624sLViqrjcbYJeg815Htsu4rmzVaZADEVC\nXhIO4MThebusdk0AcNGjgpf3HRHk0DPMDDlIjm+Oao0cqovvF6VyYmcb0C+RmhJj\ndodLXMNmbqErwTk3zEkW0yZvNIYXH7m9SokPCZa4eeIM7be62X6h1mbt0/IU6Th+\nv18fS0iTMP/Viug5und+05C/v04kgDo0CPphAbXwWMnkE4B6Tl9sdyUYXtvQsL7x\n0+WP1gL27ANqNZiI07Kz/BhbBAQI/+2TFT7oGr0AnFPQ5jHp+3GpUf6OKuT1wT3H\nND189UFuRuubxb42vZhpcXRbqJVWnbECTKVUPsGZqat3enQUB63uM4i6/RdONDZA\nfDeF1m4qYs+cUXKNUZ03\n=X6RT\n-----END PGP SIGNATURE-----')
        self.maxDiff = None
        self.assertEqual(b'tree d80c186a03f423a81b39df39dc87fd269736ca86\nparent ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd\nparent 4cffe90e0a41ad3f5190079d7c8f036bde29cbe6\nauthor James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ncommitter James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ngpgsig -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1\n \n iQIcBAABCgAGBQJULCdfAAoJEACAbyvXKaRXuKwP/RyP9PA49uAvu8tQVCC/uBa8\n vi975+xvO14R8Pp8k2nps7lSxCdtCd+xVT1VRHs0wNhOZo2YCVoU1HATkPejqSeV\n NScTHcxnk4/+bxyfk14xvJkNp7FlQ3npmBkA+lbV0Ubr33rvtIE5jiJPyz+SgWAg\n xdBG2TojV0squj00GoH/euK6aX7GgZtwdtpTv44haCQdSuPGDcI4TORqR6YSqvy3\n GPE+3ZqXPFFb+KILtimkxitdwB7CpwmNse2vE3rONSwTvi8nq3ZoQYNY73CQGkUy\n qoFU0pDtw87U3niFin1ZccDgH0bB6624sLViqrjcbYJeg815Htsu4rmzVaZADEVC\n XhIO4MThebusdk0AcNGjgpf3HRHk0DPMDDlIjm+Oao0cqovvF6VyYmcb0C+RmhJj\n dodLXMNmbqErwTk3zEkW0yZvNIYXH7m9SokPCZa4eeIM7be62X6h1mbt0/IU6Th+\n v18fS0iTMP/Viug5und+05C/v04kgDo0CPphAbXwWMnkE4B6Tl9sdyUYXtvQsL7x\n 0+WP1gL27ANqNZiI07Kz/BhbBAQI/+2TFT7oGr0AnFPQ5jHp+3GpUf6OKuT1wT3H\n ND189UFuRuubxb42vZhpcXRbqJVWnbECTKVUPsGZqat3enQUB63uM4i6/RdONDZA\n fDeF1m4qYs+cUXKNUZ03\n =X6RT\n -----END PGP SIGNATURE-----\n\nMerge ../b\n', commit.as_raw_string())

    def test_serialize_mergetag(self):
        tag = make_object(Tag, object=(Commit, b'a38d6181ff27824c79fc7df825164a212eff6a3f'), object_type_name=b'commit', name=b'v2.6.22-rc7', tag_time=1183319674, tag_timezone=0, tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org>', message=default_message)
        commit = self.make_commit(mergetag=[tag])
        self.assertEqual(b'tree d80c186a03f423a81b39df39dc87fd269736ca86\nparent ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd\nparent 4cffe90e0a41ad3f5190079d7c8f036bde29cbe6\nauthor James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ncommitter James Westby <jw+debian@jameswestby.net> 1174773719 +0000\nmergetag object a38d6181ff27824c79fc7df825164a212eff6a3f\n type commit\n tag v2.6.22-rc7\n tagger Linus Torvalds <torvalds@woody.linux-foundation.org> 1183319674 +0000\n \n Linux 2.6.22-rc7\n -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1.4.7 (GNU/Linux)\n \n iD8DBQBGiAaAF3YsRnbiHLsRAitMAKCiLboJkQECM/jpYsY3WPfvUgLXkACgg3ql\n OK2XeQOiEeXtT76rV4t2WR4=\n =ivrA\n -----END PGP SIGNATURE-----\n\nMerge ../b\n', commit.as_raw_string())

    def test_serialize_mergetags(self):
        tag = make_object(Tag, object=(Commit, b'a38d6181ff27824c79fc7df825164a212eff6a3f'), object_type_name=b'commit', name=b'v2.6.22-rc7', tag_time=1183319674, tag_timezone=0, tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org>', message=default_message)
        commit = self.make_commit(mergetag=[tag, tag])
        self.assertEqual(b'tree d80c186a03f423a81b39df39dc87fd269736ca86\nparent ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd\nparent 4cffe90e0a41ad3f5190079d7c8f036bde29cbe6\nauthor James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ncommitter James Westby <jw+debian@jameswestby.net> 1174773719 +0000\nmergetag object a38d6181ff27824c79fc7df825164a212eff6a3f\n type commit\n tag v2.6.22-rc7\n tagger Linus Torvalds <torvalds@woody.linux-foundation.org> 1183319674 +0000\n \n Linux 2.6.22-rc7\n -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1.4.7 (GNU/Linux)\n \n iD8DBQBGiAaAF3YsRnbiHLsRAitMAKCiLboJkQECM/jpYsY3WPfvUgLXkACgg3ql\n OK2XeQOiEeXtT76rV4t2WR4=\n =ivrA\n -----END PGP SIGNATURE-----\nmergetag object a38d6181ff27824c79fc7df825164a212eff6a3f\n type commit\n tag v2.6.22-rc7\n tagger Linus Torvalds <torvalds@woody.linux-foundation.org> 1183319674 +0000\n \n Linux 2.6.22-rc7\n -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1.4.7 (GNU/Linux)\n \n iD8DBQBGiAaAF3YsRnbiHLsRAitMAKCiLboJkQECM/jpYsY3WPfvUgLXkACgg3ql\n OK2XeQOiEeXtT76rV4t2WR4=\n =ivrA\n -----END PGP SIGNATURE-----\n\nMerge ../b\n', commit.as_raw_string())

    def test_deserialize_mergetag(self):
        tag = make_object(Tag, object=(Commit, b'a38d6181ff27824c79fc7df825164a212eff6a3f'), object_type_name=b'commit', name=b'v2.6.22-rc7', tag_time=1183319674, tag_timezone=0, tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org>', message=default_message)
        commit = self.make_commit(mergetag=[tag])
        d = Commit()
        d._deserialize(commit.as_raw_chunks())
        self.assertEqual(commit, d)

    def test_deserialize_mergetags(self):
        tag = make_object(Tag, object=(Commit, b'a38d6181ff27824c79fc7df825164a212eff6a3f'), object_type_name=b'commit', name=b'v2.6.22-rc7', tag_time=1183319674, tag_timezone=0, tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org>', message=default_message)
        commit = self.make_commit(mergetag=[tag, tag])
        d = Commit()
        d._deserialize(commit.as_raw_chunks())
        self.assertEqual(commit, d)
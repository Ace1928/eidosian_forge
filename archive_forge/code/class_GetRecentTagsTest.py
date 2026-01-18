import datetime
import os
import re
import shutil
import tempfile
import time
import unittest
from typing import ClassVar, Dict, List, Optional, Tuple
from dulwich.contrib import release_robot
from ..repo import Repo
from ..tests.utils import make_commit, make_tag
class GetRecentTagsTest(unittest.TestCase):
    """test get recent tags."""
    test_repo = os.path.join(BASEDIR, 'dulwich_test_repo.zip')
    committer = b'Mark Mikofski <mark.mikofski@sunpowercorp.com>'
    test_tags: ClassVar[List[bytes]] = [b'v0.1a', b'v0.1']
    tag_test_data: ClassVar[Dict[bytes, Tuple[int, bytes, Optional[Tuple[int, bytes]]]]] = {test_tags[0]: (1484788003, b'3' * 40, None), test_tags[1]: (1484788314, b'1' * 40, (1484788401, b'2' * 40))}

    @classmethod
    def setUpClass(cls):
        cls.projdir = tempfile.mkdtemp()
        cls.repo = Repo.init(cls.projdir)
        obj_store = cls.repo.object_store
        cls.c1 = make_commit(id=cls.tag_test_data[cls.test_tags[0]][1], commit_time=cls.tag_test_data[cls.test_tags[0]][0], message=b'unannotated tag', author=cls.committer)
        obj_store.add_object(cls.c1)
        cls.t1 = cls.test_tags[0]
        cls.repo[b'refs/tags/' + cls.t1] = cls.c1.id
        cls.c2 = make_commit(id=cls.tag_test_data[cls.test_tags[1]][1], commit_time=cls.tag_test_data[cls.test_tags[1]][0], message=b'annotated tag', parents=[cls.c1.id], author=cls.committer)
        obj_store.add_object(cls.c2)
        cls.t2 = make_tag(cls.c2, id=cls.tag_test_data[cls.test_tags[1]][2][1], name=cls.test_tags[1], tag_time=cls.tag_test_data[cls.test_tags[1]][2][0])
        obj_store.add_object(cls.t2)
        cls.repo[b'refs/heads/master'] = cls.c2.id
        cls.repo[b'refs/tags/' + cls.t2.name] = cls.t2.id

    @classmethod
    def tearDownClass(cls):
        cls.repo.close()
        shutil.rmtree(cls.projdir)

    def test_get_recent_tags(self):
        """Test get recent tags."""
        tags = release_robot.get_recent_tags(self.projdir)
        for tag, metadata in tags:
            tag = tag.encode('utf-8')
            test_data = self.tag_test_data[tag]
            self.assertEqual(metadata[0], gmtime_to_datetime(test_data[0]))
            self.assertEqual(metadata[1].encode('utf-8'), test_data[1])
            self.assertEqual(metadata[2].encode('utf-8'), self.committer)
            tag_obj = test_data[2]
            if not tag_obj:
                continue
            self.assertEqual(metadata[3][0], gmtime_to_datetime(tag_obj[0]))
            self.assertEqual(metadata[3][1].encode('utf-8'), tag_obj[1])
            self.assertEqual(metadata[3][2].encode('utf-8'), tag)
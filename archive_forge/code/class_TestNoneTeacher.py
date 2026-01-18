from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestNoneTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:none'
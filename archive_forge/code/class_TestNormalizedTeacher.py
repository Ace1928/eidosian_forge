from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestNormalizedTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:normalized'
from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestBothTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:both'
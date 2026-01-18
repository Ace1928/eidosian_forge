from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestSelfRevisedTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'convai2:self_revised'
from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestAll1kTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'cornell_movie:double'
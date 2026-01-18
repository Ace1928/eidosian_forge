from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestNoStartTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'dailydialog:no_start'
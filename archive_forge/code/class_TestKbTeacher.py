from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestKbTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'dialog_babi_plus:KB'
from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestMultiturnTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'ubuntu:multiturn'
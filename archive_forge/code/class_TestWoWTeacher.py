from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestWoWTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_WoW_persona_topicifier'
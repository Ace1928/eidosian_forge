from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestEDTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_ED_persona_topicifier'
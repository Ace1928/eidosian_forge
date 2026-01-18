from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestConvAI2Teacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_convAI2_persona_topicifier'
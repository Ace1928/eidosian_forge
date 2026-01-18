import unittest
from traits.api import (
from traits.observation.api import (
class TestHasTraitsObservePostInit(unittest.TestCase):
    """ Test for enthought/traits#275 """

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def test_observe_post_init_true(self):
        students = [Student() for _ in range(3)]
        teacher = Teacher(students=students)
        self.assertEqual(len(teacher.student_graduate_events), 0)
        students[0].graduate = True
        self.assertEqual(len(teacher.student_graduate_events), 1)
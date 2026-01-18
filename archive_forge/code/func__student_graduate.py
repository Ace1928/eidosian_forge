import unittest
from traits.api import (
from traits.observation.api import (
@observe(trait('students', notify=True).list_items(notify=False).trait('graduate'), post_init=True)
def _student_graduate(self, event):
    self.student_graduate_events.append(event)
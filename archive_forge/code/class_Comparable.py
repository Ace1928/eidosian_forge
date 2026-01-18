import copy
import logging
from functools import total_ordering
from tornado import web
from ..utils.tasks import as_dict, get_task_by_id, iter_tasks
from ..views import BaseHandler
@total_ordering
class Comparable:
    """
    Compare two objects, one or more of which may be None.  If one of the
    values is None, the other will be deemed greater.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        try:
            return self.value < other.value
        except TypeError:
            return self.value is None
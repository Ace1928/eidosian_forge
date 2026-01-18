import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
class ExerciseListerCustomSort(ExerciseLister):
    need_sort_by_cliff = False
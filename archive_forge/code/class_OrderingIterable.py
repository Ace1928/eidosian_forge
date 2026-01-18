import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
class OrderingIterable(utils.IterableType):

    def __init__(self, collection, operator_lt, operator_gt):
        self.collection = collection
        self.operator_lt = operator_lt
        self.operator_gt = operator_gt
        self.order = []
        self.sorted = None

    def append_field(self, selector, is_ascending):
        self.order.append((selector, is_ascending))

    def __iter__(self):
        if self.sorted is None:
            self.do_sort()
        return iter(self.sorted)

    def do_sort(outer_self):

        class Comparator(object):

            @staticmethod
            def compare(left, right):
                result = 0
                for t in outer_self.order:
                    a = t[0](left)
                    b = t[0](right)
                    if outer_self.operator_lt(a, b):
                        result = -1
                    elif outer_self.operator_gt(a, b):
                        result = 1
                    else:
                        continue
                    if not t[1]:
                        result *= -1
                    break
                return result

            def __init__(self, obj):
                self.obj = obj

            def __lt__(self, other):
                return self.compare(self.obj, other.obj) < 0

            def __gt__(self, other):
                return self.compare(self.obj, other.obj) > 0

            def __eq__(self, other):
                return self.compare(self.obj, other.obj) == 0

            def __le__(self, other):
                return self.compare(self.obj, other.obj) <= 0

            def __ge__(self, other):
                return self.compare(self.obj, other.obj) >= 0

            def __ne__(self, other):
                return self.compare(self.obj, other.obj) != 0
        outer_self.sorted = sorted(outer_self.collection, key=Comparator)
import re
import operator
from fractions import Fraction
import sys
def _operator_type_policy(obj_a, obj_b, op=operator.add):
    try:
        if type(obj_a) == type(obj_b):
            return op(obj_a, obj_b)
        if type(obj_a) in [int, long]:
            return op(type(obj_b)(obj_a), obj_b)
        if type(obj_b) in [int, long]:
            return op(type(obj_a)(obj_b), obj_a)
        raise Exception
    except (TypeError, ValueError):
        print(obj_a, obj_b)
        print(type(obj_a), type(obj_b))
        raise Exception('In _operator_type_policy, cannot apply operator')
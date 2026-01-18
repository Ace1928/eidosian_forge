import re
import operator
from fractions import Fraction
import sys
def _storage_type_policy(type_a, type_b):
    assert isinstance(type_a, type)
    assert isinstance(type_b, type)
    if type_a in [int, long]:
        return type_b
    if type_b in [int, long]:
        return type_a
    if not type_a == type_b:
        print(type_a, type_b)
        raise Exception('Bad _storage_type_policy call')
    return type_a
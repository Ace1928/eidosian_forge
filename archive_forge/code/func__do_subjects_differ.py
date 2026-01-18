from __future__ import absolute_import, division, print_function
from . import utils
def _do_subjects_differ(a, b):
    sorted_a = sorted(a, key=lambda x: (x['type'], x['name']))
    sorted_b = sorted(b, key=lambda x: (x['type'], x['name']))
    return sorted_a != sorted_b
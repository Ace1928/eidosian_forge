import collections
import copy
import itertools
import random
import re
import warnings
def _string_matcher(target):

    def match(node):
        if isinstance(node, (Clade, Tree)):
            return node.name == target
        return str(node) == target
    return match
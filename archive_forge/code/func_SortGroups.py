import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SortGroups(self):
    self._properties['mainGroup']._properties['children'] = sorted(self._properties['mainGroup']._properties['children'], key=cmp_to_key(lambda x, y: x.CompareRootGroup(y)))
    for group in self._properties['mainGroup']._properties['children']:
        if not isinstance(group, PBXGroup):
            continue
        if group.Name() == 'Products':
            products = []
            for target in self._properties['targets']:
                if not isinstance(target, PBXNativeTarget):
                    continue
                product = target._properties['productReference']
                assert product in group._properties['children']
                products.append(product)
            assert len(products) == len(group._properties['children'])
            group._properties['children'] = products
        else:
            group.SortGroup()
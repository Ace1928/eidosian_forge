import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SortRemoteProductReferences(self):

    def CompareProducts(x, y, remote_products):
        x_remote = x._properties['remoteRef']._properties['remoteGlobalIDString']
        y_remote = y._properties['remoteRef']._properties['remoteGlobalIDString']
        x_index = remote_products.index(x_remote)
        y_index = remote_products.index(y_remote)
        return cmp(x_index, y_index)
    for other_pbxproject, ref_dict in self._other_pbxprojects.items():
        remote_products = []
        for target in other_pbxproject._properties['targets']:
            if not isinstance(target, PBXNativeTarget):
                continue
            remote_products.append(target._properties['productReference'])
        product_group = ref_dict['ProductGroup']
        product_group._properties['children'] = sorted(product_group._properties['children'], key=cmp_to_key(lambda x, y, rp=remote_products: CompareProducts(x, y, rp)))
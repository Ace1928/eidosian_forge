from collections import namedtuple
import re
import textwrap
import warnings
def _item_qvalue_pair_to_header_element(pair):
    item, qvalue = pair
    if qvalue == 1.0:
        element = item
    elif qvalue == 0.0:
        element = '{};q=0'.format(item)
    else:
        element = '{};q={}'.format(item, qvalue)
    return element
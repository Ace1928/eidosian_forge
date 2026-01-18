from collections import namedtuple
import re
import textwrap
import warnings
def _item_n_weight_re(item_re):
    return '(' + item_re + ')(?:' + weight_re + ')?'
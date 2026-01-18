from collections import namedtuple
import re
import textwrap
import warnings
def _list_1_or_more__compiled_re(element_re):
    return re.compile('^(?:,' + OWS_re + ')*' + element_re + '(?:' + OWS_re + ',(?:' + OWS_re + element_re + ')?)*$')
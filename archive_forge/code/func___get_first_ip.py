from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
@staticmethod
def __get_first_ip(res):
    return res[0] if isinstance(res, list) and res else res
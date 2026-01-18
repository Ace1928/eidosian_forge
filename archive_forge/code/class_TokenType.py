from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
from cmakelang import common
class TokenType(common.EnumObject):
    _id_map = {}
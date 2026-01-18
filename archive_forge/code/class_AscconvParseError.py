import ast
import re
from collections import OrderedDict
class AscconvParseError(Exception):
    """Error parsing ascconv file"""
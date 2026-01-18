import math
import re
from typing import (
import unicodedata
from .parser import Parser
def _is_id_continue(ch):
    return unicodedata.category(ch) in ('Lu', 'Ll', 'Li', 'Lt', 'Lm', 'Lo', 'Nl', 'Nd', 'Mn', 'Mc', 'Pc')
from __future__ import unicode_literals
import re
from lib2to3.pgen2 import token
from lib2to3 import fixer_base
Optional fixer that changes all unprefixed string literals "..." to b"...".

br'abcd' is a SyntaxError on Python 2 but valid on Python 3.
ur'abcd' is a SyntaxError on Python 3 but valid on Python 2.


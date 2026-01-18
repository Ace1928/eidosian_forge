import os
import re
import sys
from collections import deque
from io import StringIO
def _print_tokens(lexer):
    while 1:
        tt = lexer.get_token()
        if not tt:
            break
        print('Token: ' + repr(tt))
import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _trace_header(self):
    print('\n           B      |\n   S   F   r   O  |        Score = Fixed - Broken\n   c   i   o   t  |  R     Fixed = num tags changed incorrect -> correct\n   o   x   k   h  |  u     Broken = num tags changed correct -> incorrect\n   r   e   e   e  |  l     Other = num tags changed incorrect -> incorrect\n   e   d   n   r  |  e\n------------------+-------------------------------------------------------\n        '.rstrip())
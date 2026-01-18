import re
from Bio import File
def _chop_end_codes(line):
    """Chops lines ending with  '     1CSA  14' and the like (PRIVATE)."""
    return re.sub('\\s\\s\\s\\s+[\\w]{4}.\\s+\\d*\\Z', '', line)
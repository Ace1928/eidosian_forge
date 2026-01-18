from typing import Tuple, Union
from Bio.SearchIO._utils import getattr_str
def _str_hsp_header(self):
    """Print the alignment header info (PRIVATE)."""
    lines = []
    qid_line = f'      Query: {self.query_id} {self.query_description}'
    qid_line = qid_line[:77] + '...' if len(qid_line) > 80 else qid_line
    hid_line = f'        Hit: {self.hit_id} {self.hit_description}'
    hid_line = hid_line[:77] + '...' if len(hid_line) > 80 else hid_line
    lines.append(qid_line)
    lines.append(hid_line)
    query_start = getattr_str(self, 'query_start')
    query_end = getattr_str(self, 'query_end')
    hit_start = getattr_str(self, 'hit_start')
    hit_end = getattr_str(self, 'hit_end')
    try:
        qstrand = self.query_strand
        hstrand = self.hit_strand
    except ValueError:
        qstrand = self.query_strand_all[0]
        hstrand = self.hit_strand_all[0]
    lines.append(f'Query range: [{query_start}:{query_end}] ({qstrand!r})')
    lines.append(f'  Hit range: [{hit_start}:{hit_end}] ({hstrand!r})')
    return '\n'.join(lines)
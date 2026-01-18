import re
import warnings
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_hit_block(self):
    """Parse a hit block (PRIVATE)."""
    self.line = read_forward(self.handle)
    match = re.search(_RE_HIT_BLOCK_DESC, self.line)
    if not match:
        raise RuntimeError(f"Unexpected content in HIT_BLOCK_DESC line'{self.line}'")
    hit_data = {'hit_id': match.group(1), 'description': match.group(2).lstrip(' ;'), 'evalue': None, 'hit_start': None, 'hit_end': None, 'hit_seq': '', 'prob': None, 'query_start': None, 'query_end': None, 'query_seq': '', 'score': None}
    self.line = self.handle.readline()
    self._process_score_line(self.line, hit_data)
    while True:
        self.line = read_forward(self.handle)
        if not self.line.strip() or self.line.startswith(_END_OF_FILE_MARKER):
            self.done = True
            return hit_data
        elif re.search(_RE_HIT_BLOCK_START, self.line):
            return hit_data
        else:
            self._parse_hit_match_block(hit_data)
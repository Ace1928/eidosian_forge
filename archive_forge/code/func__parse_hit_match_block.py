import re
import warnings
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_hit_match_block(self, hit_match_data):
    """Parse a single block of hit sequence data (PRIVATE).

        Parses block such as ::

            Q ss_pred             ceecchHHHHHHHHHHHHHHHHHHHhhhhhcCCCCccc
            Q 4P79:A|PDBID|C  160 YELGPALYLGWSASLLSILGGICVFSTAAASSKEEPAT  197 (198)
            Q Consensus       160 ~~~g~sf~l~~~~~~l~~~~~~l~~~~~~~~~~~~~~~  197 (198)
                                  .++|||||++|++.++.+++++++++..+..++++..+
            T Consensus       327 ~~~GwS~~l~~~s~~l~lia~~l~~~~~~~~~~~~~~~  364 (364)
            T 5B2G_A          327 REMGASLYVGWAASGLLLLGGGLLCCSGPSSGENLYFQ  364 (364)
            T ss_dssp             EEECTHHHHHHHHHHHHHHHHHHHHCC-----------
            T ss_pred             cccchHHHHHHHHHHHHHHHHHHHHhcCCCCCCccccC

        """

    def match_is_valid(match):
        """Return True if match is not a Consensus column (PRIVATE).

            It's not possible to distinguish a sequence line from a Consensus line with
            a regexp, so need to check the ID column.
            """
        return match.group(1).strip() != 'Consensus'
    while True:
        if not self.line.strip():
            return
        match = re.match(_RE_MATCH_BLOCK_QUERY_SEQ, self.line)
        if match and match_is_valid(match):
            hit_match_data['query_seq'] += match.group(3).strip()
            if hit_match_data['query_start'] is None:
                hit_match_data['query_start'] = int(match.group(2))
            hit_match_data['query_end'] = int(match.group(4))
        else:
            match = re.match(_RE_MATCH_BLOCK_HIT_SEQ, self.line)
            if match and match_is_valid(match):
                hit_match_data['hit_seq'] += match.group(3).strip()
                if hit_match_data['hit_start'] is None:
                    hit_match_data['hit_start'] = int(match.group(2))
                hit_match_data['hit_end'] = int(match.group(4))
        self.line = self.handle.readline()
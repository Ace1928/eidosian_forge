import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _trace_update_rules(self, num_obsolete, num_new, num_unseen):
    prefix = ' ' * 18 + '|'
    print(prefix, 'Updated rule tables:')
    print(prefix, f'  - {num_obsolete} rule applications removed')
    print(prefix, f'  - {num_new} rule applications added ({num_unseen} novel)')
    print(prefix)
from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def _write_log(self):
    """Writes the population to a logfile.

        The format is::

            timestamp: generation(if available): id1,id2,id3..."""
    if self.logfile is not None:
        ids = [str(a.info['relax_id']) for a in self.pop]
        if ids != []:
            try:
                gen_nums = [c.info['key_value_pairs']['generation'] for c in self.all_cand]
                max_gen = max(gen_nums)
            except KeyError:
                max_gen = ' '
            fd = open(self.logfile, 'a')
            fd.write('{time}: {gen}: {pop}\n'.format(time=now(), pop=','.join(ids), gen=max_gen))
            fd.close()
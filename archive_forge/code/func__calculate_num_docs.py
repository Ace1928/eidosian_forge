from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import LowCorpus
def _calculate_num_docs(self):
    """Get number of documents.

        Returns
        -------
        int
            Number of documents in file.

        """
    with utils.open(self.fname, 'rb') as fin:
        result = sum((1 for _ in fin))
    return result
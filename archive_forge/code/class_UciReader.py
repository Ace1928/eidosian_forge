import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
class UciReader(MmReader):
    """Reader of UCI format for :class:`gensim.corpora.ucicorpus.UciCorpus`."""

    def __init__(self, input):
        """

        Parameters
        ----------
        input : str
            Path to file in UCI format.

        """
        logger.info('Initializing corpus reader from %s', input)
        self.input = input
        with utils.open(self.input, 'rb') as fin:
            self.num_docs = self.num_terms = self.num_nnz = 0
            try:
                self.num_docs = int(next(fin).strip())
                self.num_terms = int(next(fin).strip())
                self.num_nnz = int(next(fin).strip())
            except StopIteration:
                pass
        logger.info('accepted corpus with %i documents, %i features, %i non-zero entries', self.num_docs, self.num_terms, self.num_nnz)

    def skip_headers(self, input_file):
        """Skip headers in `input_file`.

        Parameters
        ----------
        input_file : file
            File object.

        """
        for lineno, _ in enumerate(input_file):
            if lineno == 2:
                break
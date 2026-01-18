from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import line_tokenize
class MWAPPDBCorpusReader(WordListCorpusReader):
    """
    This class is used to read the list of word pairs from the subset of lexical
    pairs of The Paraphrase Database (PPDB) XXXL used in the Monolingual Word
    Alignment (MWA) algorithm described in Sultan et al. (2014a, 2014b, 2015):

     - http://acl2014.org/acl2014/Q14/pdf/Q14-1017
     - https://www.aclweb.org/anthology/S14-2039
     - https://www.aclweb.org/anthology/S15-2027

    The original source of the full PPDB corpus can be found on
    https://www.cis.upenn.edu/~ccb/ppdb/

    :return: a list of tuples of similar lexical terms.
    """
    mwa_ppdb_xxxl_file = 'ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs'

    def entries(self, fileids=mwa_ppdb_xxxl_file):
        """
        :return: a tuple of synonym word pairs.
        """
        return [tuple(line.split('\t')) for line in line_tokenize(self.raw(fileids))]
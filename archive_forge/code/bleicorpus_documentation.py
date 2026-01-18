from __future__ import with_statement
from os import path
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
Get document corresponding to `offset`.
        Offset can be given from :meth:`~gensim.corpora.bleicorpus.BleiCorpus.save_corpus`.

        Parameters
        ----------
        offset : int
            Position of the document in the file (in bytes).

        Returns
        -------
        list of (int, float)
            Document in BoW format.

        
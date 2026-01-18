import warnings
from collections import OrderedDict
import numpy as np
from gensim import utils
class BackMappingTranslationMatrix(utils.SaveLoad):
    """Realize the BackMapping translation matrix which maps the source model's document vector
    to the target model's document vector (old model).

    BackMapping translation matrix is used to learn a mapping for two document vector spaces which we
    specify as source document vector and target document vector. The target document vectors are trained
    on a superset corpus of source document vectors; we can incrementally increase the vector in
    the old model through the BackMapping translation matrix.

    For details on use, see the tutorial notebook [3]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>> from gensim.test.test_translation_matrix import read_sentiment_docs
        >>> from gensim.models import Doc2Vec, BackMappingTranslationMatrix
        >>>
        >>> data = read_sentiment_docs(datapath("alldata-id-10.txt"))[:5]
        >>> src_model = Doc2Vec.load(datapath("small_tag_doc_5_iter50"))
        >>> dst_model = Doc2Vec.load(datapath("large_tag_doc_10_iter50"))
        >>>
        >>> model_trans = BackMappingTranslationMatrix(src_model, dst_model)
        >>> trans_matrix = model_trans.train(data)
        >>>
        >>> result = model_trans.infer_vector(dst_model.dv[data[3].tags])

    """

    def __init__(self, source_lang_vec, target_lang_vec, tagged_docs=None, random_state=None):
        """

        Parameters
        ----------
        source_lang_vec : :class:`~gensim.models.doc2vec.Doc2Vec`
            Source Doc2Vec model.
        target_lang_vec : :class:`~gensim.models.doc2vec.Doc2Vec`
            Target Doc2Vec model.
        tagged_docs : list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional.
            Documents that will be used for training, both the source language document vector and
            target language document vector trained on those tagged documents.
        random_state : {None, int, array_like}, optional
            Seed for random state.

        """
        self.tagged_docs = tagged_docs
        self.source_lang_vec = source_lang_vec
        self.target_lang_vec = target_lang_vec
        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None
        if tagged_docs is not None:
            self.train(tagged_docs)

    def train(self, tagged_docs):
        """Build the translation matrix to map from the source model's vectors to target model's vectors

        Parameters
        ----------
        tagged_docs : list of :class:`~gensim.models.doc2vec.TaggedDocument`, Documents
            that will be used for training, both the source language document vector and
            target language document vector trained on those tagged documents.

        Returns
        -------
        numpy.ndarray
            Translation matrix that maps from the source model's vectors to target model's vectors.

        """
        m1 = [self.source_lang_vec.dv[item.tags].flatten() for item in tagged_docs]
        m2 = [self.target_lang_vec.dv[item.tags].flatten() for item in tagged_docs]
        self.translation_matrix = np.linalg.lstsq(m2, m1, -1)[0]
        return self.translation_matrix

    def infer_vector(self, target_doc_vec):
        """Translate the target model's document vector to the source model's document vector

        Parameters
        ----------
        target_doc_vec : numpy.ndarray
            Document vector from the target document, whose document are not in the source model.

        Returns
        -------
        numpy.ndarray
            Vector `target_doc_vec` in the source model.

        """
        return np.dot(target_doc_vec, self.translation_matrix)
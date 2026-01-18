import warnings
from collections import OrderedDict
import numpy as np
from gensim import utils
class TranslationMatrix(utils.SaveLoad):
    """Objects of this class realize the translation matrix which maps the source language to the target language.
    The main methods are:

    We map it to the other language space by computing z = Wx, then return the
    word whose representation is close to z.

    For details on use, see the tutorial notebook [3]_

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.models import KeyedVectors
        >>> from gensim.test.utils import datapath
        >>> en = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        >>> it = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        >>> model_en = KeyedVectors.load_word2vec_format(en)
        >>> model_it = KeyedVectors.load_word2vec_format(it)
        >>>
        >>> word_pairs = [
        ...     ("one", "uno"), ("two", "due"), ("three", "tre"), ("four", "quattro"), ("five", "cinque"),
        ...     ("seven", "sette"), ("eight", "otto"),
        ...     ("dog", "cane"), ("pig", "maiale"), ("fish", "cavallo"), ("birds", "uccelli"),
        ...     ("apple", "mela"), ("orange", "arancione"), ("grape", "acino"), ("banana", "banana")
        ... ]
        >>>
        >>> trans_model = TranslationMatrix(model_en, model_it)
        >>> trans_model.train(word_pairs)
        >>> trans_model.translate(["dog", "one"], topn=3)
        OrderedDict([('dog', [u'cane', u'gatto', u'cavallo']), ('one', [u'uno', u'due', u'tre'])])


    References
    ----------
    .. [3] https://github.com/RaRe-Technologies/gensim/blob/3.2.0/docs/notebooks/translation_matrix.ipynb

    """

    def __init__(self, source_lang_vec, target_lang_vec, word_pairs=None, random_state=None):
        """
        Parameters
        ----------
        source_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`
            Word vectors for source language.
        target_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`
            Word vectors for target language.
        word_pairs : list of (str, str), optional
            Pairs of words that will be used for training.
        random_state : {None, int, array_like}, optional
            Seed for random state.

        """
        self.source_word = None
        self.target_word = None
        self.source_lang_vec = source_lang_vec
        self.target_lang_vec = target_lang_vec
        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None
        self.source_space = None
        self.target_space = None
        if word_pairs is not None:
            if len(word_pairs[0]) != 2:
                raise ValueError('Each training data item must contain two different language words.')
            self.train(word_pairs)

    def train(self, word_pairs):
        """Build the translation matrix to map from source space to target space.

        Parameters
        ----------
        word_pairs : list of (str, str), optional
            Pairs of words that will be used for training.

        """
        self.source_word, self.target_word = zip(*word_pairs)
        self.source_space = Space.build(self.source_lang_vec, set(self.source_word))
        self.target_space = Space.build(self.target_lang_vec, set(self.target_word))
        self.source_space.normalize()
        self.target_space.normalize()
        m1 = self.source_space.mat[[self.source_space.word2index[item] for item in self.source_word], :]
        m2 = self.target_space.mat[[self.target_space.word2index[item] for item in self.target_word], :]
        self.translation_matrix = np.linalg.lstsq(m1, m2, -1)[0]

    def save(self, *args, **kwargs):
        """Save the model to a file. Ignores (doesn't store) the `source_space` and `target_space` attributes."""
        kwargs['ignore'] = kwargs.get('ignore', ['source_space', 'target_space'])
        super(TranslationMatrix, self).save(*args, **kwargs)

    def apply_transmat(self, words_space):
        """Map the source word vector to the target word vector using translation matrix.

        Parameters
        ----------
        words_space : :class:`~gensim.models.translation_matrix.Space`
            `Space` object constructed for the words to be translated.

        Returns
        -------
        :class:`~gensim.models.translation_matrix.Space`
            `Space` object constructed for the mapped words.

        """
        return Space(np.dot(words_space.mat, self.translation_matrix), words_space.index2word)

    def translate(self, source_words, topn=5, gc=0, sample_num=None, source_lang_vec=None, target_lang_vec=None):
        """Translate the word from the source language to the target language.

        Parameters
        ----------
        source_words : {str, list of str}
            Single word or a list of words to be translated
        topn : int, optional
            Number of words that will be returned as translation for each `source_words`
        gc : int, optional
            Define translation algorithm, if `gc == 0` - use standard NN retrieval,
            otherwise, use globally corrected neighbour retrieval method (as described in [1]_).
        sample_num : int, optional
            Number of words to sample from the source lexicon, if `gc == 1`, then `sample_num` **must** be provided.
        source_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
            New source language vectors for translation, by default, used the model's source language vector.
        target_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
            New target language vectors for translation, by default, used the model's target language vector.

        Returns
        -------
        :class:`collections.OrderedDict`
            Ordered dict where each item is `word`: [`translated_word_1`, `translated_word_2`, ...]

        """
        if isinstance(source_words, str):
            source_words = [source_words]
        if source_lang_vec is None:
            warnings.warn("The parameter source_lang_vec isn't specified, use the model's source language word vector as default.")
            source_lang_vec = self.source_lang_vec
        if target_lang_vec is None:
            warnings.warn("The parameter target_lang_vec isn't specified, use the model's target language word vector as default.")
            target_lang_vec = self.target_lang_vec
        if gc:
            if sample_num is None:
                raise RuntimeError('When using the globally corrected neighbour retrieval method, the `sample_num` parameter(i.e. the number of words sampled from source space) must be provided.')
            lexicon = set(source_lang_vec.index_to_key)
            addition = min(sample_num, len(lexicon) - len(source_words))
            lexicon = self.random_state.choice(list(lexicon.difference(source_words)), addition)
            source_space = Space.build(source_lang_vec, set(source_words).union(set(lexicon)))
        else:
            source_space = Space.build(source_lang_vec, source_words)
        target_space = Space.build(target_lang_vec)
        source_space.normalize()
        target_space.normalize()
        mapped_source_space = self.apply_transmat(source_space)
        sim_matrix = -np.dot(target_space.mat, mapped_source_space.mat.T)
        if gc:
            srtd_idx = np.argsort(np.argsort(sim_matrix, axis=1), axis=1)
            sim_matrix_idx = np.argsort(srtd_idx + sim_matrix, axis=0)
        else:
            sim_matrix_idx = np.argsort(sim_matrix, axis=0)
        translated_word = OrderedDict()
        for idx, word in enumerate(source_words):
            translated_target_word = []
            for j in range(topn):
                map_space_id = sim_matrix_idx[j, source_space.word2index[word]]
                translated_target_word.append(target_space.index2word[map_space_id])
            translated_word[word] = translated_target_word
        return translated_word
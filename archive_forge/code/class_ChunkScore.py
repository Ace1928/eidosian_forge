import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
class ChunkScore:
    """
    A utility class for scoring chunk parsers.  ``ChunkScore`` can
    evaluate a chunk parser's output, based on a number of statistics
    (precision, recall, f-measure, misssed chunks, incorrect chunks).
    It can also combine the scores from the parsing of multiple texts;
    this makes it significantly easier to evaluate a chunk parser that
    operates one sentence at a time.

    Texts are evaluated with the ``score`` method.  The results of
    evaluation can be accessed via a number of accessor methods, such
    as ``precision`` and ``f_measure``.  A typical use of the
    ``ChunkScore`` class is::

        >>> chunkscore = ChunkScore()           # doctest: +SKIP
        >>> for correct in correct_sentences:   # doctest: +SKIP
        ...     guess = chunkparser.parse(correct.leaves())   # doctest: +SKIP
        ...     chunkscore.score(correct, guess)              # doctest: +SKIP
        >>> print('F Measure:', chunkscore.f_measure())       # doctest: +SKIP
        F Measure: 0.823

    :ivar kwargs: Keyword arguments:

        - max_tp_examples: The maximum number actual examples of true
          positives to record.  This affects the ``correct`` member
          function: ``correct`` will not return more than this number
          of true positive examples.  This does *not* affect any of
          the numerical metrics (precision, recall, or f-measure)

        - max_fp_examples: The maximum number actual examples of false
          positives to record.  This affects the ``incorrect`` member
          function and the ``guessed`` member function: ``incorrect``
          will not return more than this number of examples, and
          ``guessed`` will not return more than this number of true
          positive examples.  This does *not* affect any of the
          numerical metrics (precision, recall, or f-measure)

        - max_fn_examples: The maximum number actual examples of false
          negatives to record.  This affects the ``missed`` member
          function and the ``correct`` member function: ``missed``
          will not return more than this number of examples, and
          ``correct`` will not return more than this number of true
          negative examples.  This does *not* affect any of the
          numerical metrics (precision, recall, or f-measure)

        - chunk_label: A regular expression indicating which chunks
          should be compared.  Defaults to ``'.*'`` (i.e., all chunks).

    :type _tp: list(Token)
    :ivar _tp: List of true positives
    :type _fp: list(Token)
    :ivar _fp: List of false positives
    :type _fn: list(Token)
    :ivar _fn: List of false negatives

    :type _tp_num: int
    :ivar _tp_num: Number of true positives
    :type _fp_num: int
    :ivar _fp_num: Number of false positives
    :type _fn_num: int
    :ivar _fn_num: Number of false negatives.
    """

    def __init__(self, **kwargs):
        self._correct = set()
        self._guessed = set()
        self._tp = set()
        self._fp = set()
        self._fn = set()
        self._max_tp = kwargs.get('max_tp_examples', 100)
        self._max_fp = kwargs.get('max_fp_examples', 100)
        self._max_fn = kwargs.get('max_fn_examples', 100)
        self._chunk_label = kwargs.get('chunk_label', '.*')
        self._tp_num = 0
        self._fp_num = 0
        self._fn_num = 0
        self._count = 0
        self._tags_correct = 0.0
        self._tags_total = 0.0
        self._measuresNeedUpdate = False

    def _updateMeasures(self):
        if self._measuresNeedUpdate:
            self._tp = self._guessed & self._correct
            self._fn = self._correct - self._guessed
            self._fp = self._guessed - self._correct
            self._tp_num = len(self._tp)
            self._fp_num = len(self._fp)
            self._fn_num = len(self._fn)
            self._measuresNeedUpdate = False

    def score(self, correct, guessed):
        """
        Given a correctly chunked sentence, score another chunked
        version of the same sentence.

        :type correct: chunk structure
        :param correct: The known-correct ("gold standard") chunked
            sentence.
        :type guessed: chunk structure
        :param guessed: The chunked sentence to be scored.
        """
        self._correct |= _chunksets(correct, self._count, self._chunk_label)
        self._guessed |= _chunksets(guessed, self._count, self._chunk_label)
        self._count += 1
        self._measuresNeedUpdate = True
        try:
            correct_tags = tree2conlltags(correct)
            guessed_tags = tree2conlltags(guessed)
        except ValueError:
            correct_tags = guessed_tags = ()
        self._tags_total += len(correct_tags)
        self._tags_correct += sum((1 for t, g in zip(guessed_tags, correct_tags) if t == g))

    def accuracy(self):
        """
        Return the overall tag-based accuracy for all text that have
        been scored by this ``ChunkScore``, using the IOB (conll2000)
        tag encoding.

        :rtype: float
        """
        if self._tags_total == 0:
            return 1
        return self._tags_correct / self._tags_total

    def precision(self):
        """
        Return the overall precision for all texts that have been
        scored by this ``ChunkScore``.

        :rtype: float
        """
        self._updateMeasures()
        div = self._tp_num + self._fp_num
        if div == 0:
            return 0
        else:
            return self._tp_num / div

    def recall(self):
        """
        Return the overall recall for all texts that have been
        scored by this ``ChunkScore``.

        :rtype: float
        """
        self._updateMeasures()
        div = self._tp_num + self._fn_num
        if div == 0:
            return 0
        else:
            return self._tp_num / div

    def f_measure(self, alpha=0.5):
        """
        Return the overall F measure for all texts that have been
        scored by this ``ChunkScore``.

        :param alpha: the relative weighting of precision and recall.
            Larger alpha biases the score towards the precision value,
            while smaller alpha biases the score towards the recall
            value.  ``alpha`` should have a value in the range [0,1].
        :type alpha: float
        :rtype: float
        """
        self._updateMeasures()
        p = self.precision()
        r = self.recall()
        if p == 0 or r == 0:
            return 0
        return 1 / (alpha / p + (1 - alpha) / r)

    def missed(self):
        """
        Return the chunks which were included in the
        correct chunk structures, but not in the guessed chunk
        structures, listed in input order.

        :rtype: list of chunks
        """
        self._updateMeasures()
        chunks = list(self._fn)
        return [c[1] for c in chunks]

    def incorrect(self):
        """
        Return the chunks which were included in the guessed chunk structures,
        but not in the correct chunk structures, listed in input order.

        :rtype: list of chunks
        """
        self._updateMeasures()
        chunks = list(self._fp)
        return [c[1] for c in chunks]

    def correct(self):
        """
        Return the chunks which were included in the correct
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
        chunks = list(self._correct)
        return [c[1] for c in chunks]

    def guessed(self):
        """
        Return the chunks which were included in the guessed
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
        chunks = list(self._guessed)
        return [c[1] for c in chunks]

    def __len__(self):
        self._updateMeasures()
        return self._tp_num + self._fn_num

    def __repr__(self):
        """
        Return a concise representation of this ``ChunkScoring``.

        :rtype: str
        """
        return '<ChunkScoring of ' + repr(len(self)) + ' chunks>'

    def __str__(self):
        """
        Return a verbose representation of this ``ChunkScoring``.
        This representation includes the precision, recall, and
        f-measure scores.  For other information about the score,
        use the accessor methods (e.g., ``missed()`` and ``incorrect()``).

        :rtype: str
        """
        return 'ChunkParse score:\n' + f'    IOB Accuracy: {self.accuracy() * 100:5.1f}%%\n' + f'    Precision:    {self.precision() * 100:5.1f}%%\n' + f'    Recall:       {self.recall() * 100:5.1f}%%\n' + f'    F-Measure:    {self.f_measure() * 100:5.1f}%%'
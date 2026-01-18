from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash, simple_split_tokenizer
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class HashingVectorizer(Preprocessor):
    """Count the frequency of tokens using the
    `hashing trick <https://en.wikipedia.org/wiki/Feature_hashing>`_.

    This preprocessors creates ``num_features`` columns named like
    ``hash_{column_name}_{index}``. If ``num_features`` is large enough relative to
    the size of your vocabulary, then each column approximately corresponds to the
    frequency of a unique token.

    :class:`HashingVectorizer` is memory efficient and quick to pickle. However, given a
    transformed column, you can't know which tokens correspond to it. This might make it
    hard to determine which tokens are important to your model.

    .. note::

        This preprocessor transforms each input column to a
        `document-term matrix <https://en.wikipedia.org/wiki/Document-term_matrix>`_.

        A document-term matrix is a table that describes the frequency of tokens in a
        collection of documents. For example, the strings `"I like Python"` and `"I
        dislike Python"` might have the document-term matrix below:

        .. code-block::

                    corpus_I  corpus_Python  corpus_dislike  corpus_like
                0         1              1               1            0
                1         1              1               0            1

        To generate the matrix, you typically map each token to a unique index. For
        example:

        .. code-block::

                        token  index
                0        I      0
                1   Python      1
                2  dislike      2
                3     like      3

        The problem with this approach is that memory use scales linearly with the size
        of your vocabulary. :class:`HashingVectorizer` circumvents this problem by
        computing indices with a hash function:
        :math:`\\texttt{index} = hash(\\texttt{token})`.

    .. warning::
        Sparse matrices aren't currently supported. If you use a large ``num_features``,
        this preprocessor might behave poorly.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import HashingVectorizer
        >>>
        >>> df = pd.DataFrame({
        ...     "corpus": [
        ...         "Jimmy likes volleyball",
        ...         "Bob likes volleyball too",
        ...         "Bob also likes fruit jerky"
        ...     ]
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>>
        >>> vectorizer = HashingVectorizer(["corpus"], num_features=8)
        >>> vectorizer.fit_transform(ds).to_pandas()  # doctest: +SKIP
           hash_corpus_0  hash_corpus_1  hash_corpus_2  hash_corpus_3  hash_corpus_4  hash_corpus_5  hash_corpus_6  hash_corpus_7
        0              1              0              1              0              0              0              0              1
        1              1              0              1              0              0              0              1              1
        2              0              0              1              1              0              2              1              0

    Args:
        columns: The columns to separately tokenize and count.
        num_features: The number of features used to represent the vocabulary. You
            should choose a value large enough to prevent hash collisions between
            distinct tokens.
        tokenization_fn: The function used to generate tokens. This function
            should accept a string as input and return a list of tokens as
            output. If unspecified, the tokenizer uses a function equivalent to
            ``lambda s: s.split(" ")``.

    .. seealso::

        :class:`CountVectorizer`
            Another method for counting token frequencies. Unlike :class:`HashingVectorizer`,
            :class:`CountVectorizer` creates a feature for each unique token. This
            enables you to compute the inverse transformation.

        :class:`FeatureHasher`
            This preprocessor is similar to :class:`HashingVectorizer`, except it expects
            a table describing token frequencies. In contrast,
            :class:`FeatureHasher` expects a column containing documents.
    """
    _is_fittable = False

    def __init__(self, columns: List[str], num_features: int, tokenization_fn: Optional[Callable[[str], List[str]]]=None):
        self.columns = columns
        self.num_features = num_features
        self.tokenization_fn = tokenization_fn or simple_split_tokenizer

    def _transform_pandas(self, df: pd.DataFrame):

        def hash_count(tokens: List[str]) -> Counter:
            hashed_tokens = [simple_hash(token, self.num_features) for token in tokens]
            return Counter(hashed_tokens)
        for col in self.columns:
            tokenized = df[col].map(self.tokenization_fn)
            hashed = tokenized.map(hash_count)
            for i in range(self.num_features):
                df[f'hash_{col}_{i}'] = hashed.map(lambda counts: counts[i])
        df.drop(columns=self.columns, inplace=True)
        return df

    def __repr__(self):
        fn_name = getattr(self.tokenization_fn, '__name__', self.tokenization_fn)
        return f'{self.__class__.__name__}(columns={self.columns!r}, num_features={self.num_features!r}, tokenization_fn={fn_name})'
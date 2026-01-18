import array
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..exceptions import NotFittedError
from ..preprocessing import normalize
from ..utils import _IS_32BIT
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils.validation import FLOAT_DTYPES, check_array, check_is_fitted
from ._hash import FeatureHasher
from ._stop_words import ENGLISH_STOP_WORDS
class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    _white_spaces = re.compile('\\s\\s+')

    def decode(self, doc):
        """Decode the input into a string of unicode symbols.

        The decoding strategy depends on the vectorizer parameters.

        Parameters
        ----------
        doc : bytes or str
            The string to decode.

        Returns
        -------
        doc: str
            A string of unicode symbols.
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()
        elif self.input == 'file':
            doc = doc.read()
        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)
        if doc is np.nan:
            raise ValueError('np.nan is an invalid document, expected byte or unicode string.')
        return doc

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []
            n_original_tokens = len(original_tokens)
            tokens_append = tokens.append
            space_join = ' '.join
            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i:i + n]))
        return tokens

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        text_document = self._white_spaces.sub(' ', text_document)
        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []
        ngrams_append = ngrams.append
        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i:i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
        operating only inside word boundaries. n-grams at the edges
        of words are padded with space."""
        text_document = self._white_spaces.sub(' ', text_document)
        min_n, max_n = self.ngram_range
        ngrams = []
        ngrams_append = ngrams.append
        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:
                    break
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization.

        Returns
        -------
        preprocessor: callable
              A function to preprocess the text before tokenization.
        """
        if self.preprocessor is not None:
            return self.preprocessor
        if not self.strip_accents:
            strip_accents = None
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' % self.strip_accents)
        return partial(_preprocess, accent_function=strip_accents, lower=self.lowercase)

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens.

        Returns
        -------
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        if token_pattern.groups > 1:
            raise ValueError('More than 1 capturing group in token pattern. Only a single group should be captured.')
        return token_pattern.findall

    def get_stop_words(self):
        """Build or fetch the effective stop words list.

        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        return _check_stop_list(self.stop_words)

    def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
        """Check if stop words are consistent

        Returns
        -------
        is_consistent : True if stop words are consistent with the preprocessor
                        and tokenizer, False if they are not, None if the check
                        was previously performed, "error" if it could not be
                        performed (e.g. because of the use of a custom
                        preprocessor / tokenizer)
        """
        if id(self.stop_words) == getattr(self, '_stop_words_id', None):
            return None
        try:
            inconsistent = set()
            for w in stop_words or ():
                tokens = list(tokenize(preprocess(w)))
                for token in tokens:
                    if token not in stop_words:
                        inconsistent.add(token)
            self._stop_words_id = id(self.stop_words)
            if inconsistent:
                warnings.warn('Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens %r not in stop_words.' % sorted(inconsistent))
            return not inconsistent
        except Exception:
            self._stop_words_id = id(self.stop_words)
            return 'error'

    def build_analyzer(self):
        """Return a callable to process input data.

        The callable handles preprocessing, tokenization, and n-grams generation.

        Returns
        -------
        analyzer: callable
            A function to handle preprocessing, tokenization
            and n-grams generation.
        """
        if callable(self.analyzer):
            return partial(_analyze, analyzer=self.analyzer, decoder=self.decode)
        preprocess = self.build_preprocessor()
        if self.analyzer == 'char':
            return partial(_analyze, ngrams=self._char_ngrams, preprocessor=preprocess, decoder=self.decode)
        elif self.analyzer == 'char_wb':
            return partial(_analyze, ngrams=self._char_wb_ngrams, preprocessor=preprocess, decoder=self.decode)
        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            self._check_stop_words_consistency(stop_words, preprocess, tokenize)
            return partial(_analyze, ngrams=self._word_ngrams, tokenizer=tokenize, preprocessor=preprocess, decoder=self.decode, stop_words=stop_words)
        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = 'Duplicate term in vocabulary: %r' % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(vocabulary.values())
                if len(indices) != len(vocabulary):
                    raise ValueError('Vocabulary contains repeated indices.')
                for i in range(len(vocabulary)):
                    if i not in indices:
                        msg = "Vocabulary of size %d doesn't contain index %d." % (len(vocabulary), i)
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError('empty vocabulary passed to fit')
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fitted)"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
            if not self.fixed_vocabulary_:
                raise NotFittedError('Vocabulary not fitted or provided')
        if len(self.vocabulary_) == 0:
            raise ValueError('Vocabulary is empty')

    def _validate_ngram_range(self):
        """Check validity of ngram_range parameter"""
        min_n, max_m = self.ngram_range
        if min_n > max_m:
            raise ValueError('Invalid value for ngram_range=%s lower boundary larger than the upper boundary.' % str(self.ngram_range))

    def _warn_for_unused_params(self):
        if self.tokenizer is not None and self.token_pattern is not None:
            warnings.warn("The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")
        if self.preprocessor is not None and callable(self.analyzer):
            warnings.warn("The parameter 'preprocessor' will not be used since 'analyzer' is callable'")
        if self.ngram_range != (1, 1) and self.ngram_range is not None and callable(self.analyzer):
            warnings.warn("The parameter 'ngram_range' will not be used since 'analyzer' is callable'")
        if self.analyzer != 'word' or callable(self.analyzer):
            if self.stop_words is not None:
                warnings.warn("The parameter 'stop_words' will not be used since 'analyzer' != 'word'")
            if self.token_pattern is not None and self.token_pattern != '(?u)\\b\\w\\w+\\b':
                warnings.warn("The parameter 'token_pattern' will not be used since 'analyzer' != 'word'")
            if self.tokenizer is not None:
                warnings.warn("The parameter 'tokenizer' will not be used since 'analyzer' != 'word'")
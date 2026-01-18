from __future__ import with_statement
import logging
import os
import random
import re
import sys
from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import (
from gensim.utils import deaccent, simple_tokenize
from smart_open import open
class TextCorpus(interfaces.CorpusABC):
    """Helper class to simplify the pipeline of getting BoW vectors from plain text.

    Notes
    -----
    This is an abstract base class: override the :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` and
    :meth:`~gensim.corpora.textcorpus.TextCorpus.__len__` methods to match your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object will be automatically initialized
    with a dictionary in `self.dictionary` and will support the :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__`
    corpus method.  You have a few different ways of utilizing this class via subclassing or by construction with
    different preprocessing arguments.

    The :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__` method converts the lists of tokens produced by
    :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` to BoW format using
    :meth:`gensim.corpora.dictionary.Dictionary.doc2bow`.

    :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` does the following:

    #. Calls :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream` to get a generator over the texts.
       It yields each document in turn from the underlying text file or files.
    #. For each document from the stream, calls :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` to produce
       a list of tokens. If metadata=True, it yields a 2-`tuple` with the document number as the second element.

    Preprocessing consists of 0+ `character_filters`, a `tokenizer`, and 0+ `token_filters`.

    The preprocessing consists of calling each filter in `character_filters` with the document text.
    Unicode is not guaranteed, and if desired, the first filter should convert to unicode.
    The output of each character filter should be another string. The output from the final filter is fed
    to the `tokenizer`, which should split the string into a list of tokens (strings).
    Afterwards, the list of tokens is fed through each filter in `token_filters`. The final output returned from
    :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` is the output from the final token filter.

    So to use this class, you can either pass in different preprocessing functions using the
    `character_filters`, `tokenizer`, and `token_filters` arguments, or you can subclass it.

    If subclassing: override :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream` to take text from different input
    sources in different formats.
    Override :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` if you must provide different initial
    preprocessing, then call the :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` method to apply
    the normal preprocessing.
    You can also override :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` in order to tag the documents
    (token lists) with different metadata.

    The default preprocessing consists of:

    #. :func:`~gensim.parsing.preprocessing.lower_to_unicode` - lowercase and convert to unicode (assumes utf8 encoding)
    #. :func:`~gensim.utils.deaccent`- deaccent (asciifolding)
    #. :func:`~gensim.parsing.preprocessing.strip_multiple_whitespaces` - collapse multiple whitespaces into one
    #. :func:`~gensim.utils.simple_tokenize` - tokenize by splitting on whitespace
    #. :func:`~gensim.parsing.preprocessing.remove_short_tokens` - remove words less than 3 characters long
    #. :func:`~gensim.parsing.preprocessing.remove_stopword_tokens` - remove stopwords

    """

    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None, tokenizer=None, token_filters=None):
        """

        Parameters
        ----------
        input : str, optional
            Path to top-level directory (file) to traverse for corpus documents.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.
            If `input` is None, the dictionary will remain uninitialized.
        metadata : bool, optional
            If True - yield metadata with each document.
        character_filters : iterable of callable, optional
            Each will be applied to the text of each document in order, and should return a single string with
            the modified text. For Python 2, the original text will not be unicode, so it may be useful to
            convert to unicode as the first character filter.
            If None - using :func:`~gensim.parsing.preprocessing.lower_to_unicode`,
            :func:`~gensim.utils.deaccent` and :func:`~gensim.parsing.preprocessing.strip_multiple_whitespaces`.
        tokenizer : callable, optional
            Tokenizer for document, if None - using :func:`~gensim.utils.simple_tokenize`.
        token_filters : iterable of callable, optional
            Each will be applied to the iterable of tokens in order, and should return another iterable of tokens.
            These filters can add, remove, or replace tokens, or do nothing at all.
            If None - using :func:`~gensim.parsing.preprocessing.remove_short_tokens` and
            :func:`~gensim.parsing.preprocessing.remove_stopword_tokens`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim import utils
            >>>
            >>>
            >>> class CorpusMiislita(TextCorpus):
            ...     stopwords = set('for a of the and to in on'.split())
            ...
            ...     def get_texts(self):
            ...         for doc in self.getstream():
            ...             yield [word for word in utils.to_unicode(doc).lower().split() if word not in self.stopwords]
            ...
            ...     def __len__(self):
            ...         self.length = sum(1 for _ in self.get_texts())
            ...         return self.length
            >>>
            >>>
            >>> corpus = CorpusMiislita(datapath('head500.noblanks.cor.bz2'))
            >>> len(corpus)
            250
            >>> document = next(iter(corpus.get_texts()))

        """
        self.input = input
        self.metadata = metadata
        self.character_filters = character_filters
        if self.character_filters is None:
            self.character_filters = [lower_to_unicode, deaccent, strip_multiple_whitespaces]
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = simple_tokenize
        self.token_filters = token_filters
        if self.token_filters is None:
            self.token_filters = [remove_short_tokens, remove_stopword_tokens]
        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary)

    def init_dictionary(self, dictionary):
        """Initialize/update dictionary.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.

        Notes
        -----
        If self.input is None - make nothing.

        """
        self.dictionary = dictionary if dictionary is not None else Dictionary()
        if self.input is not None:
            if dictionary is None:
                logger.info('Initializing dictionary')
                metadata_setting = self.metadata
                self.metadata = False
                self.dictionary.add_documents(self.get_texts())
                self.metadata = metadata_setting
            else:
                logger.info('Input stream provided but dictionary already initialized')
        else:
            logger.warning('No input document stream provided; assuming dictionary will be initialized some other way.')

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format (+ metadata if self.metadata).

        """
        if self.metadata:
            for text, metadata in self.get_texts():
                yield (self.dictionary.doc2bow(text, allow_update=False), metadata)
        else:
            for text in self.get_texts():
                yield self.dictionary.doc2bow(text, allow_update=False)

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            Document read from plain-text file.

        Notes
        -----
        After generator end - initialize self.length attribute.

        """
        num_texts = 0
        with utils.file_or_filename(self.input) as f:
            for line in f:
                yield line
                num_texts += 1
        self.length = num_texts

    def preprocess_text(self, text):
        """Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.

        Parameters
        ---------
        text : str
            Document read from plain-text file.

        Return
        ------
        list of str
            List of tokens extracted from `text`.

        """
        for character_filter in self.character_filters:
            text = character_filter(text)
        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)
        return tokens

    def step_through_preprocess(self, text):
        """Apply preprocessor one by one and generate result.

        Warnings
        --------
        This is useful for debugging issues with the corpus preprocessing pipeline.

        Parameters
        ----------
        text : str
            Document text read from plain-text file.

        Yields
        ------
        (callable, object)
            Pre-processor, output from pre-processor (based on `text`)

        """
        for character_filter in self.character_filters:
            text = character_filter(text)
            yield (character_filter, text)
        tokens = self.tokenizer(text)
        yield (self.tokenizer, tokens)
        for token_filter in self.token_filters:
            yield (token_filter, token_filter(tokens))

    def get_texts(self):
        """Generate documents from corpus.

        Yields
        ------
        list of str
            Document as sequence of tokens (+ lineno if self.metadata)

        """
        lines = self.getstream()
        if self.metadata:
            for lineno, line in enumerate(lines):
                yield (self.preprocess_text(line), (lineno,))
        else:
            for line in lines:
                yield self.preprocess_text(line)

    def sample_texts(self, n, seed=None, length=None):
        """Generate `n` random documents from the corpus without replacement.

        Parameters
        ----------
        n : int
            Number of documents we want to sample.
        seed : int, optional
            If specified, use it as a seed for local random generator.
        length : int, optional
            Value will used as corpus length (because calculate length of corpus can be costly operation).
            If not specified - will call `__length__`.

        Raises
        ------
        ValueError
            If `n` less than zero or greater than corpus size.

        Notes
        -----
        Given the number of remaining documents in a corpus, we need to choose n elements.
        The probability for the current element to be chosen is `n` / remaining. If we choose it,  we just decrease
        the `n` and move to the next element.

        Yields
        ------
        list of str
            Sampled document as sequence of tokens.

        """
        random_generator = random if seed is None else random.Random(seed)
        if length is None:
            length = len(self)
        if not n <= length:
            raise ValueError('n {0:d} is larger/equal than length of corpus {1:d}.'.format(n, length))
        if not 0 <= n:
            raise ValueError('Negative sample size n {0:d}.'.format(n))
        i = 0
        for i, sample in enumerate(self.getstream()):
            if i == length:
                break
            remaining_in_corpus = length - i
            chance = random_generator.randint(1, remaining_in_corpus)
            if chance <= n:
                n -= 1
                if self.metadata:
                    yield (self.preprocess_text(sample[0]), sample[1])
                else:
                    yield self.preprocess_text(sample)
        if n != 0:
            raise ValueError('length {0:d} greater than number of documents in corpus {1:d}'.format(length, i + 1))

    def __len__(self):
        """Get length of corpus

        Warnings
        --------
        If self.length is None - will read all corpus for calculate this attribute through
        :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream`.

        Returns
        -------
        int
            Length of corpus.

        """
        if self.length is None:
            self.length = sum((1 for _ in self.getstream()))
        return self.length
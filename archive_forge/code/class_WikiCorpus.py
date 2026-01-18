import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
class WikiCorpus(TextCorpus):
    """Treat a Wikipedia articles dump as a read-only, streamed, memory-efficient corpus.

    Supported dump formats:

    * <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    * <LANG>wiki-latest-pages-articles.xml.bz2

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    Notes
    -----
    Dumps for the English Wikipedia can be founded at https://dumps.wikimedia.org/enwiki/.

    Attributes
    ----------
    metadata : bool
        Whether to write articles titles to serialized corpus.

    Warnings
    --------
    "Multistream" archives are *not* supported in Python 2 due to `limitations in the core bz2 library
    <https://docs.python.org/2/library/bz2.html#de-compression-of-files>`_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> from gensim.corpora import WikiCorpus, MmCorpus
        >>>
        >>> path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
        >>> corpus_path = get_tmpfile("wiki-corpus.mm")
        >>>
        >>> wiki = WikiCorpus(path_to_wiki_dump)  # create word->word_id mapping, ~8h on full wiki
        >>> MmCorpus.serialize(corpus_path, wiki)  # another 8h, creates a file in MatrixMarket format and mapping

    """

    def __init__(self, fname, processes=None, lemmatize=None, dictionary=None, metadata=False, filter_namespaces=('0',), tokenizer_func=tokenize, article_min_tokens=ARTICLE_MIN_WORDS, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True, filter_articles=None):
        """Initialize the corpus.

        Unless a dictionary is provided, this scans the corpus once,
        to determine its vocabulary.

        Parameters
        ----------
        fname : str
            Path to the Wikipedia dump file.
        processes : int, optional
            Number of processes to run, defaults to `max(1, number of cpu - 1)`.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Dictionary, if not provided,  this scans the corpus once, to determine its vocabulary
            **IMPORTANT: this needs a really long time**.
        filter_namespaces : tuple of str, optional
            Namespaces to consider.
        tokenizer_func : function, optional
            Function that will be used for tokenization. By default, use :func:`~gensim.corpora.wikicorpus.tokenize`.
            If you inject your own tokenizer, it must conform to this interface:
            `tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str`
        article_min_tokens : int, optional
            Minimum tokens in article. Article will be ignored if number of tokens is less.
        token_min_len : int, optional
            Minimal token length.
        token_max_len : int, optional
            Maximal token length.
        lower : bool, optional
             If True - convert all text to lower case.
        filter_articles: callable or None, optional
            If set, each XML article element will be passed to this callable before being processed. Only articles
            where the callable returns an XML element are processed, returning None allows filtering out
            some articles based on customised rules.
        metadata: bool
            Have the `get_texts()` method yield `(content_tokens, (page_id, page_title))` tuples, instead
            of just `content_tokens`.

        Warnings
        --------
        Unless a dictionary is provided, this scans the corpus once, to determine its vocabulary.

        """
        if lemmatize is not None:
            raise NotImplementedError('The lemmatize parameter is no longer supported. If you need to lemmatize, use e.g. <https://github.com/clips/pattern>. Perform lemmatization as part of your tokenization function and pass it as the tokenizer_func parameter to this initializer.')
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.filter_articles = filter_articles
        self.metadata = metadata
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.tokenizer_func = tokenizer_func
        self.article_min_tokens = article_min_tokens
        self.token_min_len = token_min_len
        self.token_max_len = token_max_len
        self.lower = lower
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary

    @property
    def input(self):
        return self.fname

    def get_texts(self):
        """Iterate over the dump, yielding a list of tokens for each article that passed
        the length and namespace filtering.

        Uses multiprocessing internally to parallelize the work and process the dump more quickly.

        Notes
        -----
        This iterates over the **texts**. If you want vectors, just use the standard corpus interface
        instead of this method:

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.corpora import WikiCorpus
            >>>
            >>> path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
            >>>
            >>> for vec in WikiCorpus(path_to_wiki_dump):
            ...     pass

        Yields
        ------
        list of str
            If `metadata` is False, yield only list of token extracted from the article.
        (list of str, (int, str))
            List of tokens (extracted from the article), page id and article title otherwise.

        """
        articles, articles_all = (0, 0)
        positions, positions_all = (0, 0)
        tokenization_params = (self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
        texts = ((text, title, pageid, tokenization_params) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces, self.filter_articles))
        pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)
        try:
            for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
                for tokens, title, pageid in pool.imap(_process_article, group):
                    articles_all += 1
                    positions_all += len(tokens)
                    if len(tokens) < self.article_min_tokens or any((title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES)):
                        continue
                    articles += 1
                    positions += len(tokens)
                    if self.metadata:
                        yield (tokens, (pageid, title))
                    else:
                        yield tokens
        except KeyboardInterrupt:
            logger.warning('user terminated iteration over Wikipedia corpus after %i documents with %i positions (total %i articles, %i positions before pruning articles shorter than %i words)', articles, positions, articles_all, positions_all, self.article_min_tokens)
        except PicklingError as exc:
            raise PicklingError(f'Can not send filtering function {self.filter_articles} to multiprocessing, make sure the function can be pickled.') from exc
        else:
            logger.info('finished iterating over Wikipedia corpus of %i documents with %i positions (total %i articles, %i positions before pruning articles shorter than %i words)', articles, positions, articles_all, positions_all, self.article_min_tokens)
            self.length = articles
        finally:
            pool.terminate()
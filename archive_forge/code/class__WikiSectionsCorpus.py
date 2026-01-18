import argparse
import json
import logging
import multiprocessing
import re
import sys
from xml.etree import ElementTree
from functools import partial
from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, WikiCorpus, filter_wiki, find_interlinks, get_namespace, utils
import gensim.utils
class _WikiSectionsCorpus(WikiCorpus):
    """Treat a wikipedia articles dump (<LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    or <LANG>wiki-latest-pages-articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    """

    def __init__(self, fileobj, min_article_character=200, processes=None, lemmatize=None, filter_namespaces=('0',), include_interlinks=False):
        """
        Parameters
        ----------
        fileobj : file
            File descriptor of MediaWiki dump.
        min_article_character : int, optional
            Minimal number of character for article (except titles and leading gaps).
        processes : int, optional
            Number of processes, max(1, multiprocessing.cpu_count() - 1) if None.
        filter_namespaces : tuple of int, optional
            Enumeration of namespaces that will be ignored.
        include_interlinks: bool
            Whether or not interlinks should be included in the output

        """
        if lemmatize is not None:
            raise NotImplementedError('The lemmatize parameter is no longer supported since Gensim 4.0.0. If you need to lemmatize, use e.g. https://github.com/clips/pattern to preprocess your corpus before submitting it to Gensim.')
        self.fileobj = fileobj
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.min_article_character = min_article_character
        self.include_interlinks = include_interlinks

    def get_texts_with_sections(self):
        """Iterate over the dump, returning titles and text versions of all sections of articles.

        Notes
        -----
        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function:

        .. sourcecode:: pycon

            >>> for vec in wiki_corpus:
            >>>     print(vec)

        Yields
        ------
        (str, list of (str, str), list of (str, str))
            Structure contains (title, [(section_heading, section_content), ...],
            (Optionally)[(interlink_article, interlink_text), ...]).

        """
        skipped_namespace, skipped_length, skipped_redirect = (0, 0, 0)
        total_articles, total_sections = (0, 0)
        page_xmls = extract_page_xmls(self.fileobj)
        pool = multiprocessing.Pool(self.processes)
        for group in utils.chunkize(page_xmls, chunksize=10 * self.processes, maxsize=1):
            for article in pool.imap(partial(segment, include_interlinks=self.include_interlinks), group):
                article_title, sections = (article[0], article[1])
                if any((article_title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES)):
                    skipped_namespace += 1
                    continue
                if not sections or sections[0][1].lstrip().lower().startswith('#redirect'):
                    skipped_redirect += 1
                    continue
                if sum((len(body.strip()) for _, body in sections)) < self.min_article_character:
                    skipped_length += 1
                    continue
                total_articles += 1
                total_sections += len(sections)
                if self.include_interlinks:
                    interlinks = article[2]
                    yield (article_title, sections, interlinks)
                else:
                    yield (article_title, sections)
        logger.info('finished processing %i articles with %i sections (skipped %i redirects, %i stubs, %i ignored namespaces)', total_articles, total_sections, skipped_redirect, skipped_length, skipped_namespace)
        pool.terminate()
        self.length = total_articles
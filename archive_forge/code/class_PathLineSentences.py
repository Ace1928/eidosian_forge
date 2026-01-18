from the disk or network on-the-fly, without loading your entire corpus into RAM.
from __future__ import division  # py3 "true division"
import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from types import GeneratorType
import threading
import itertools
import copy
from queue import Queue, Empty
from numpy import float32 as REAL
import numpy as np
from gensim.utils import keep_vocab_item, call_on_class_only, deprecated
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
from gensim import utils, matutils
from gensim.models.keyedvectors import Vocab  # noqa
from smart_open.compression import get_supported_extensions
class PathLineSentences:

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory
        in alphabetical order by filename.

        The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
        .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

        The format of files (either text, or compressed text files) in the path is one sentence = one line,
        with words already preprocessed and separated by whitespace.

        Warnings
        --------
        Does **not recurse** into subdirectories.

        Parameters
        ----------
        source : str
            Path to the directory.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]
            self.input_files.sort()
        else:
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with utils.open(file_name, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length
from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
class TestTextDirectoryCorpus(unittest.TestCase):

    def write_one_level(self, *args):
        if not args:
            args = ('doc1', 'doc2')
        dirpath = tempfile.mkdtemp()
        self.write_docs_to_directory(dirpath, *args)
        return dirpath

    def write_docs_to_directory(self, dirpath, *args):
        for doc_num, name in enumerate(args):
            with open(os.path.join(dirpath, name), 'w') as f:
                f.write('document %d content' % doc_num)

    def test_one_level_directory(self):
        dirpath = self.write_one_level()
        corpus = textcorpus.TextDirectoryCorpus(dirpath)
        self.assertEqual(len(corpus), 2)
        docs = list(corpus)
        self.assertEqual(len(docs), 2)

    def write_two_levels(self):
        dirpath = self.write_one_level()
        next_level = os.path.join(dirpath, 'level_two')
        os.mkdir(next_level)
        self.write_docs_to_directory(next_level, 'doc1', 'doc2')
        return (dirpath, next_level)

    def test_two_level_directory(self):
        dirpath, next_level = self.write_two_levels()
        corpus = textcorpus.TextDirectoryCorpus(dirpath)
        self.assertEqual(len(corpus), 4)
        docs = list(corpus)
        self.assertEqual(len(docs), 4)
        corpus = textcorpus.TextDirectoryCorpus(dirpath, min_depth=1)
        self.assertEqual(len(corpus), 2)
        docs = list(corpus)
        self.assertEqual(len(docs), 2)
        corpus = textcorpus.TextDirectoryCorpus(dirpath, max_depth=0)
        self.assertEqual(len(corpus), 2)
        docs = list(corpus)
        self.assertEqual(len(docs), 2)

    def test_filename_filtering(self):
        dirpath = self.write_one_level('test1.log', 'test1.txt', 'test2.log', 'other1.log')
        corpus = textcorpus.TextDirectoryCorpus(dirpath, pattern='test.*\\.log')
        filenames = list(corpus.iter_filepaths())
        expected = [os.path.join(dirpath, name) for name in ('test1.log', 'test2.log')]
        self.assertEqual(sorted(expected), sorted(filenames))
        corpus.pattern = '.*.txt'
        filenames = list(corpus.iter_filepaths())
        expected = [os.path.join(dirpath, 'test1.txt')]
        self.assertEqual(expected, filenames)
        corpus.pattern = None
        corpus.exclude_pattern = '.*.log'
        filenames = list(corpus.iter_filepaths())
        self.assertEqual(expected, filenames)

    def test_lines_are_documents(self):
        dirpath = tempfile.mkdtemp()
        lines = ['doc%d text' % i for i in range(5)]
        fpath = os.path.join(dirpath, 'test_file.txt')
        with open(fpath, 'w') as f:
            f.write('\n'.join(lines))
        corpus = textcorpus.TextDirectoryCorpus(dirpath, lines_are_documents=True)
        docs = [doc for doc in corpus.getstream()]
        self.assertEqual(len(lines), corpus.length)
        self.assertEqual(lines, docs)
        corpus.lines_are_documents = False
        docs = [doc for doc in corpus.getstream()]
        self.assertEqual(1, corpus.length)
        self.assertEqual('\n'.join(lines), docs[0])

    def test_non_trivial_structure(self):
        """Test with non-trivial directory structure, shown below:
        .
        ├── 0.txt
        ├── a_folder
        │   └── 1.txt
        └── b_folder
            ├── 2.txt
            ├── 3.txt
            └── c_folder
                └── 4.txt
        """
        dirpath = tempfile.mkdtemp()
        self.write_docs_to_directory(dirpath, '0.txt')
        a_folder = os.path.join(dirpath, 'a_folder')
        os.mkdir(a_folder)
        self.write_docs_to_directory(a_folder, '1.txt')
        b_folder = os.path.join(dirpath, 'b_folder')
        os.mkdir(b_folder)
        self.write_docs_to_directory(b_folder, '2.txt', '3.txt')
        c_folder = os.path.join(b_folder, 'c_folder')
        os.mkdir(c_folder)
        self.write_docs_to_directory(c_folder, '4.txt')
        corpus = textcorpus.TextDirectoryCorpus(dirpath)
        filenames = list(corpus.iter_filepaths())
        base_names = sorted((name[len(dirpath) + 1:] for name in filenames))
        expected = sorted(['0.txt', 'a_folder/1.txt', 'b_folder/2.txt', 'b_folder/3.txt', 'b_folder/c_folder/4.txt'])
        expected = [os.path.normpath(path) for path in expected]
        self.assertEqual(expected, base_names)
        corpus.max_depth = 1
        self.assertEqual(expected[:-1], base_names[:-1])
        corpus.min_depth = 1
        self.assertEqual(expected[2:-1], base_names[2:-1])
        corpus.max_depth = 0
        self.assertEqual(expected[2:], base_names[2:])
        corpus.pattern = '4.*'
        self.assertEqual(expected[-1], base_names[-1])
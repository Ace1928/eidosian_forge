import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
class TransfoXLCorpus(object):

    @classmethod
    @torch_only_method
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a pre-processed corpus.
        """
        vocab = TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        try:
            resolved_corpus_file = cached_file(pretrained_model_name_or_path, CORPUS_NAME, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(f"Corpus '{pretrained_model_name_or_path}' was not found in corpus list ({', '.join(PRETRAINED_CORPUS_ARCHIVE_MAP.keys())}. We assumed '{pretrained_model_name_or_path}' was a path or url but couldn't find files {CORPUS_NAME} at this path or url.")
            return None
        if is_local:
            logger.info(f'loading corpus file {resolved_corpus_file}')
        else:
            logger.info(f'loading corpus file {CORPUS_NAME} from cache at {resolved_corpus_file}')
        corpus = cls(*inputs, **kwargs)
        corpus_dict = torch.load(resolved_corpus_file)
        for key, value in corpus_dict.items():
            corpus.__dict__[key] = value
        corpus.vocab = vocab
        if corpus.train is not None:
            corpus.train = torch.tensor(corpus.train, dtype=torch.long)
        if corpus.valid is not None:
            corpus.valid = torch.tensor(corpus.valid, dtype=torch.long)
        if corpus.test is not None:
            corpus.test = torch.tensor(corpus.test, dtype=torch.long)
        return corpus

    def __init__(self, *args, **kwargs):
        self.vocab = TransfoXLTokenizer(*args, **kwargs)
        self.dataset = None
        self.train = None
        self.valid = None
        self.test = None

    def build_corpus(self, path, dataset):
        self.dataset = dataset
        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(path, '1-billion-word-language-modeling-benchmark-r13output', 'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
        self.vocab.build_vocab()
        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(os.path.join(path, 'train.txt'), ordered=True)
            self.valid = self.vocab.encode_file(os.path.join(path, 'valid.txt'), ordered=True)
            self.test = self.vocab.encode_file(os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)
        else:
            data_iter = None
            raise ValueError(f'Split not recognized: {split}')
        return data_iter
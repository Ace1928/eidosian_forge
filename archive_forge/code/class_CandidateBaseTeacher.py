from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
class CandidateBaseTeacher(Teacher, ABC):
    """
    Base Teacher.

    Contains some functions that are useful for all the subteachers.
    """

    def __init__(self, opt: Opt, shared: dict=None, vocab_size: int=VOCAB_SIZE, example_size: int=EXAMPLE_SIZE, num_candidates: int=NUM_CANDIDATES, num_train: int=NUM_TRAIN, num_test: int=NUM_TEST):
        """
        :param int vocab_size:
            size of the vocabulary
        :param int example_size:
            length of each example
        :param int num_candidates:
            number of label_candidates generated
        :param int num_train:
            size of the training set
        :param int num_test:
            size of the valid/test sets
        """
        self.opt = opt
        opt['datafile'] = opt['datatype'].split(':')[0]
        self.datafile = opt['datafile']
        self.vocab_size = vocab_size
        self.example_size = example_size
        self.num_candidates = num_candidates
        self.num_train = num_train
        self.num_test = num_test
        self.words = list(map(str, range(self.vocab_size)))
        super().__init__(opt, shared)

    def build_corpus(self):
        """
        Build corpus; override for customization.
        """
        return [list(x) for x in itertools.permutations(self.words, self.example_size)]

    def num_episodes(self) -> int:
        if self.datafile == 'train':
            return self.num_train
        else:
            return self.num_test

    def num_examples(self) -> int:
        return self.num_episodes()

    def _setup_data(self, fold: str):
        self.rng = random.Random(42)
        full_corpus = self.build_corpus()
        self.rng.shuffle(full_corpus)
        it = iter(full_corpus)
        self.train = list(itertools.islice(it, self.num_train))
        self.val = list(itertools.islice(it, self.num_test))
        self.test = list(itertools.islice(it, self.num_test))
        assert len(self.train) == self.num_train, len(self.train)
        assert len(self.val) == self.num_test, len(self.val)
        assert len(self.test) == self.num_test, len(self.test)
        assert len(set(itertools.chain(*self.train)) - set(self.words)) == 0
        if fold == 'train':
            self.corpus = self.train
        elif fold == 'valid':
            self.corpus = self.val
        elif fold == 'test':
            self.corpus = self.test
        self.corpus = [' '.join(x) for x in self.corpus]
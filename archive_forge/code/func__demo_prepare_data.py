import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def _demo_prepare_data(tagged_data, train, num_sents, randomize, separate_baseline_data):
    if tagged_data is None:
        print('Loading tagged data from treebank... ')
        tagged_data = treebank.tagged_sents()
    if num_sents is None or len(tagged_data) <= num_sents:
        num_sents = len(tagged_data)
    if randomize:
        random.seed(len(tagged_data))
        random.shuffle(tagged_data)
    cutoff = int(num_sents * train)
    training_data = tagged_data[:cutoff]
    gold_data = tagged_data[cutoff:num_sents]
    testing_data = [[t[0] for t in sent] for sent in gold_data]
    if not separate_baseline_data:
        baseline_data = training_data
    else:
        bl_cutoff = len(training_data) // 3
        baseline_data, training_data = (training_data[:bl_cutoff], training_data[bl_cutoff:])
    trainseqs, traintokens = corpus_size(training_data)
    testseqs, testtokens = corpus_size(testing_data)
    bltrainseqs, bltraintokens = corpus_size(baseline_data)
    print(f'Read testing data ({testseqs:d} sents/{testtokens:d} wds)')
    print(f'Read training data ({trainseqs:d} sents/{traintokens:d} wds)')
    print('Read baseline data ({:d} sents/{:d} wds) {:s}'.format(bltrainseqs, bltraintokens, '' if separate_baseline_data else '[reused the training set]'))
    return (training_data, baseline_data, gold_data, testing_data)
import logging
import pickle
import random
from collections import defaultdict
from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI
def _get_pretrain_model():
    tagger = PerceptronTagger()
    training = _load_data_conll_format('english_ptb_train.conll')
    testing = _load_data_conll_format('english_ptb_test.conll')
    print('Size of training and testing (sentence)', len(training), len(testing))
    tagger.train(training, PICKLE)
    print('Accuracy : ', tagger.accuracy(testing))
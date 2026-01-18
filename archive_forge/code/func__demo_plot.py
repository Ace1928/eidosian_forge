import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def _demo_plot(learning_curve_output, teststats, trainstats=None, take=None):
    testcurve = [teststats['initialerrors']]
    for rulescore in teststats['rulescores']:
        testcurve.append(testcurve[-1] - rulescore)
    testcurve = [1 - x / teststats['tokencount'] for x in testcurve[:take]]
    traincurve = [trainstats['initialerrors']]
    for rulescore in trainstats['rulescores']:
        traincurve.append(traincurve[-1] - rulescore)
    traincurve = [1 - x / trainstats['tokencount'] for x in traincurve[:take]]
    import matplotlib.pyplot as plt
    r = list(range(len(testcurve)))
    plt.plot(r, testcurve, r, traincurve)
    plt.axis([None, None, None, 1.0])
    plt.savefig(learning_curve_output)
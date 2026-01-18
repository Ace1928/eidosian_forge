import os
import tempfile
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file
from nltk.classify.tadm import call_tadm, parse_tadm_weights, write_tadm_file
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.data import gzip_open_unicode
from nltk.probability import DictionaryProbDist
from nltk.util import OrderedDict
class TadmMaxentClassifier(MaxentClassifier):

    @classmethod
    def train(cls, train_toks, **kwargs):
        algorithm = kwargs.get('algorithm', 'tao_lmvm')
        trace = kwargs.get('trace', 3)
        encoding = kwargs.get('encoding', None)
        labels = kwargs.get('labels', None)
        sigma = kwargs.get('gaussian_prior_sigma', 0)
        count_cutoff = kwargs.get('count_cutoff', 0)
        max_iter = kwargs.get('max_iter')
        ll_delta = kwargs.get('min_lldelta')
        if not encoding:
            encoding = TadmEventMaxentFeatureEncoding.train(train_toks, count_cutoff, labels=labels)
        trainfile_fd, trainfile_name = tempfile.mkstemp(prefix='nltk-tadm-events-', suffix='.gz')
        weightfile_fd, weightfile_name = tempfile.mkstemp(prefix='nltk-tadm-weights-')
        trainfile = gzip_open_unicode(trainfile_name, 'w')
        write_tadm_file(train_toks, encoding, trainfile)
        trainfile.close()
        options = []
        options.extend(['-monitor'])
        options.extend(['-method', algorithm])
        if sigma:
            options.extend(['-l2', '%.6f' % sigma ** 2])
        if max_iter:
            options.extend(['-max_it', '%d' % max_iter])
        if ll_delta:
            options.extend(['-fatol', '%.6f' % abs(ll_delta)])
        options.extend(['-events_in', trainfile_name])
        options.extend(['-params_out', weightfile_name])
        if trace < 3:
            options.extend(['2>&1'])
        else:
            options.extend(['-summary'])
        call_tadm(options)
        with open(weightfile_name) as weightfile:
            weights = parse_tadm_weights(weightfile)
        os.remove(trainfile_name)
        os.remove(weightfile_name)
        weights *= numpy.log2(numpy.e)
        return cls(encoding, weights)
from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def _check_old_version(self, old_version):
    logging.info('TESTING LOAD of %s Doc2Vec MODEL', old_version)
    saved_models_dir = datapath('old_d2v_models/d2v_{}.mdl')
    model = doc2vec.Doc2Vec.load(saved_models_dir.format(old_version))
    self.assertTrue(len(model.wv) == 3)
    self.assertIsNone(model.corpus_total_words)
    self.assertTrue(model.wv.vectors.shape == (3, 4))
    self.assertTrue(model.dv.vectors.shape == (2, 4))
    self.assertTrue(len(model.dv) == 2)
    doc0_inferred = model.infer_vector(list(DocsLeeCorpus())[0].words)
    sims_to_infer = model.dv.most_similar([doc0_inferred], topn=len(model.dv))
    self.assertTrue(sims_to_infer)
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    model.save(tmpf)
    loaded_model = doc2vec.Doc2Vec.load(tmpf)
    doc0_inferred = loaded_model.infer_vector(list(DocsLeeCorpus())[0].words)
    sims_to_infer = loaded_model.dv.most_similar([doc0_inferred], topn=len(loaded_model.dv))
    self.assertTrue(sims_to_infer)
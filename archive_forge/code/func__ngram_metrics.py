import json
import math
import os
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from nltk import ngrams
from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import AverageMetric, SumMetric, GlobalAverageMetric
from parlai.utils.misc import round_sigfigs
def _ngram_metrics(self, batch, preds):
    text_vecs_cpu = batch.text_vec.cpu()
    lrep, crep = (0, 0)
    total_pred_ngs = 0
    n = self.opt['seq_ul_n']
    for i, pred in enumerate(preds):
        pred_token_list = pred.tolist()
        if self.END_IDX in pred_token_list:
            pred_token_list = pred_token_list[:pred_token_list.index(self.END_IDX)]
        if self.START_IDX in pred_token_list:
            pred_token_list = pred_token_list[pred_token_list.index(self.START_IDX):]
        pred_ngs = [ng for ng in ngrams(pred_token_list, n)]
        pred_counter = Counter(pred_ngs)
        total_pred_ngs += len(pred_ngs)
        lrep += len(pred_ngs) - len(pred_counter)
        text_token_list = text_vecs_cpu[i].tolist()
        if self.NULL_IDX in text_token_list:
            text_token_list = text_token_list[:text_token_list.index(self.NULL_IDX)]
        context_counter = Counter([ng for ng in ngrams(text_token_list, n)])
        for ng in pred_counter:
            if ng in context_counter:
                crep += pred_counter[ng]
    self.global_metrics.add('lrep_%dgrams' % n, GlobalAverageMetric(lrep, total_pred_ngs))
    self.global_metrics.add('crep_%dgrams' % n, GlobalAverageMetric(crep, total_pred_ngs))
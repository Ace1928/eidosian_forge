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
class _VocabUnlikelihoodTrait(object):

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['high_freq'] = 0
        self.metrics['gold_high'] = 0

    def _kldiv(self, p_counter, q_counter) -> float:
        ptotal = sum(p_counter.values())
        qtotal = sum(q_counter.values())
        kldiv = 0.0
        for word, _ in p_counter.items():
            prob_p = p_counter[word] / ptotal
            prob_q = q_counter[word] / qtotal
            kldiv += prob_p * math.log(1e-20 + prob_q / prob_p)
        return -kldiv

    def _jsdiv(self, dist1: Counter, dist2: Counter) -> float:
        half = dist1 + dist2
        return 0.5 * self._kldiv(dist1, half) + 0.5 * self._kldiv(dist2, half)

    def report(self):
        report = super().report()
        report['kldiv_humgen'] = self._kldiv(self.running_human, self.running_generation)
        report['kldiv_genhum'] = self._kldiv(self.running_generation, self.running_human)
        report['jsdiv'] = self._jsdiv(self.running_human, self.running_generation)
        return report
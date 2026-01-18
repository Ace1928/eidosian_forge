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
class SequenceVocabUnlikelihoodAgentTrait(_VocabUnlikelihoodTrait):
    """
    Abstract Trait.

    Applies unlikelihood loss to vocabulary distributiion by generating, calculating
    proportion of tokens per vocabulary frequency bin, and computing loss accordingly
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.NUM_STEPS = opt['queue_size']
        if shared is None:
            self._reset_running_histories()
            self._last_was_training = True
            self.truebins = {}
            counts_file = self.opt['counts_file']
            if counts_file is None:
                counts_file = os.path.join(os.path.dirname(self.opt['model_file']), 'counts.txt')
                if not os.path.isfile(counts_file):
                    raise RuntimeError('Please give a --counts-file to use vocab unlikelihood')
            with open(counts_file) as f:
                for line in f:
                    record = json.loads(line)
                    self.truebins[record['word_id']] = record['bin']

    def reset(self):
        super().reset()
        self._reset_running_histories()

    def _reset_running_histories(self):
        self.generation_history = []
        self.running_generation = Counter()
        self.human_history = []
        self.running_human = Counter()

    @classmethod
    def add_cmdline_args(cls, argparser):
        print(super())
        grp = super().add_cmdline_args(argparser)
        grp.add_argument('--alpha', default=1.0, type=float)
        grp.add_argument('--queue-size', default=32, type=int)
        grp.add_argument('--weighting', choices={'uniform', 'logdiff', 'kldiv'}, default='uniform')
        grp.add_argument('--threshold', type=float, default=0.001)
        grp.add_argument('--counts-file', type=str, default=None)

    def _init_cuda_buffer(self, *args, **kwargs):
        pass

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['num_penalize'] = 0
        self.metrics['steps'] = 0
        self.metrics['hum_toks'] = 0
        self.metrics['gen_toks'] = 0
        self.metrics['ul_weights'] = 0

    def _get_bins(self, counts: Counter):
        c = Counter()
        for k, v in counts.items():
            c.update({self.truebins.get(k, 'never'): v})
        t = sum(c.values())
        return {k: round_sigfigs(v / t, 4) for k, v in c.items()}

    def _l2dist(self, bins):
        return (bins.get('frequent', 0) - 0.4) ** 2 + (bins.get('medium', 0) - 0.3) ** 2 + (bins.get('rare', 0) - 0.2) ** 2 + (bins.get('veryrare', 0) - 0.1) ** 2 + (bins.get('never', 0) - 0.0) ** 2

    def report(self):
        r = super().report()
        if self.running_generation and self.running_human:
            for k, v in self._get_bins(self.running_human).items():
                r[f'humdist_{k}'] = v
            gendist = self._get_bins(self.running_generation)
            for k, v in gendist.items():
                r[f'gendist_{k}'] = v
            r['dist_l2'] = self._l2dist(gendist)
        return r

    def compute_loss(self, batch, return_output=False):
        if self._last_was_training is not self.is_training:
            self._reset_running_histories()
            self._last_was_training = self.is_training
        nll_loss, model_output = super().compute_loss(batch, True)
        scores, preds, *_ = model_output
        targets = batch.label_vec
        notnull = targets != self.NULL_IDX
        with torch.no_grad():
            beam_pred_scores, _ = self._generate(batch, self.beam_size, self.opt['label_truncate'])
            generations = [g for g, s in beam_pred_scores]
            gentoks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True, padding_value=self.NULL_IDX)
            gentoks = gentoks[:, 1:]
        gen_mask = gentoks != self.NULL_IDX
        self.generation_history.append(Counter(gentoks[gen_mask].view(-1).tolist()))
        self.human_history.append(Counter(targets[notnull].view(-1).tolist()))
        self.running_generation += self.generation_history[-1]
        self.running_human += self.human_history[-1]
        if len(self.generation_history) > self.NUM_STEPS:
            if not self.is_training:
                self.running_generation -= self.generation_history.pop(0)
                self.running_human -= self.human_history.pop(0)
        elif return_output:
            return (nll_loss, model_output)
        else:
            return nll_loss
        gen_sum = sum(self.running_generation.values())
        hum_sum = sum(self.running_human.values())
        if self.opt['weighting'] == 'logdiff':
            to_penalize = {w: v / gen_sum - self.running_human.get(w, 0) / hum_sum for w, v in self.running_generation.items()}
            to_penalize = {w: v for w, v in to_penalize.items() if v >= self.opt['threshold']}
            to_penalize = {w: math.log(v / 0.001) for w, v in to_penalize.items()}
        elif self.opt['weighting'] == 'uniform':
            to_penalize = {w: v / gen_sum - self.running_human.get(w, 0) / hum_sum for w, v in self.running_generation.items()}
            to_penalize = {w: 1 for w, v in to_penalize.items() if v >= self.opt['threshold']}
        elif self.opt['weighting'] == 'kldiv':
            to_penalize = {w: (self.running_generation[w] / gen_sum, self.running_human[w] / hum_sum) for w, v in self.running_human.items() if w in self.running_generation}
            to_penalize = {w: (p_gen, p_hum) for w, (p_gen, p_hum) in to_penalize.items() if p_gen > p_hum}
            to_penalize = {w: p_gen * (math.log(p_gen) - math.log(p_hum)) for w, (p_gen, p_hum) in to_penalize.items()}
            to_penalize = {k: v for k, v in to_penalize.items() if v > self.opt['threshold']}
        else:
            raise ValueError
        self.global_metrics.add('num_penalize', SumMetric(len(to_penalize)))
        ul_weights = torch.zeros(gen_mask.shape)
        ul_mask = torch.zeros_like(gen_mask)
        for wordid, weight in to_penalize.items():
            ul_mask = ul_mask | (gentoks == wordid)
            ul_weights[gentoks == wordid] = weight
        ul_weights = ul_weights.to(gen_mask.device)
        self.global_metrics.add('ul_weights', AverageMetric(ul_weights[ul_mask].mean()))
        model_output = self.model(*self._model_input(batch), ys=gentoks)
        scores, *_ = model_output
        downweight = gentoks[ul_mask]
        almost_scores = F.log_softmax(scores[ul_mask], dim=-1)
        ul_scores = almost_scores[torch.arange(len(downweight)), downweight]
        clamp_min = 1e-06 if self.opt['fp16'] else 1e-20
        ul_loss = (-torch.log(torch.clamp(1 - ul_scores.exp(), min=clamp_min)) * ul_weights[ul_mask]).sum()
        num_ul = ul_mask.sum()
        self.global_metrics.add('ul_loss', AverageMetric(ul_loss, num_ul))
        self.global_metrics.add('ul_num_tokens', SumMetric(num_ul))
        ul_loss = div(ul_loss, num_ul)
        if len(self.generation_history) < self.NUM_STEPS:
            loss = nll_loss
        else:
            loss = nll_loss + self.opt['alpha'] * ul_loss
        if return_output:
            return (loss, model_output)
        else:
            return loss
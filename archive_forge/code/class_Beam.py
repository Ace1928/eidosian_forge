from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
class Beam(object):
    """
    Generic beam class.

    It keeps information about beam_size hypothesis.
    """

    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1, eos_token=2, min_n_best=3, cuda='cpu', block_ngram=0):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param min_length:
            minimum length of the predicted sequence
        :param padding_token:
            Set to 0 as usual in ParlAI
        :param bos_token:
            Set to 1 as usual in ParlAI
        :param eos_token:
            Set to 2 as usual in ParlAI
        :param min_n_best:
            Beam will not be done unless this amount of finished hypothesis
            (with EOS) is done
        :param cuda:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(self.device)
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        self.bookkeep = []
        self.outputs = [torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)]
        self.finished = []
        self.HypothesisTail = namedtuple('HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid'])
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.block_ngram = block_ngram
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    @staticmethod
    def find_ngrams(input_list, n):
        """
        Get list of ngrams with context length n-1.
        """
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_output_from_current_step(self):
        """
        Get the outputput at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        """
        Advance the beam one step.
        """
        voc_size = softmax_probs.size(-1)
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            for hyp_id in range(softmax_probs.size(0)):
                softmax_probs[hyp_id][self.eos] = neginf(softmax_probs.dtype)
        if len(self.bookkeep) == 0:
            beam_scores = softmax_probs[0]
        else:
            beam_scores = softmax_probs + self.scores.unsqueeze(1).expand_as(softmax_probs)
            for i in range(self.outputs[-1].size(0)):
                if self.block_ngram > 0:
                    current_hypo = self.partial_hyps[i][1:]
                    current_ngrams = []
                    for ng in range(self.block_ngram):
                        ngrams = Beam.find_ngrams(current_hypo, ng)
                        if len(ngrams) > 0:
                            current_ngrams.extend(ngrams)
                    counted_ngrams = Counter(current_ngrams)
                    if any((v > 1 for k, v in counted_ngrams.items())):
                        beam_scores[i] = neginf(softmax_probs.dtype)
                if self.outputs[-1][i] == self.eos:
                    beam_scores[i] = neginf(softmax_probs.dtype)
        flatten_beam_scores = beam_scores.view(-1)
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(flatten_beam_scores, self.beam_size, dim=-1)
        self.scores = best_scores
        self.all_scores.append(self.scores)
        hyp_ids = best_idxs / voc_size
        tok_ids = best_idxs % voc_size
        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()] for i in range(self.beam_size)]
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                eostail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=hypid, score=self.scores[hypid], tokenid=self.eos)
                self.finished.append(eostail)
                self.n_best_counter += 1
        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        """
        Return whether beam search is complete.
        """
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Get single best hypothesis.

        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (self.get_hyp_from_finished(top_hypothesis_tail), top_hypothesis_tail.score)

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs)-1
        :param hyp_id:
            id with range up to beam_size-1
        :return:
            hypothesis sequence
        """
        assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback], tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]
        return hyp_idx

    @staticmethod
    def get_pretty_hypothesis(list_of_hypotails):
        """
        Return prettier version of the hypotheses.
        """
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)
        hypothesis = torch.stack(list(reversed(hypothesis)))
        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses in rescored order.

        :param n_best:
            how many n best hypothesis to return
        :return:
            list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            length_penalty = math.pow((1 + current_length) / 6, 0.65)
            rescored_finished.append(self.HypothesisTail(timestep=finished_item.timestep, hypid=finished_item.hypid, score=finished_item.score / length_penalty, tokenid=finished_item.tokenid))
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)
        if n_best is not None:
            srted = srted[:n_best]
        return srted

    def check_finished(self):
        """
        Check if self.finished is empty and add hyptail in that case.

        This will be suboptimal hypothesis since the model did not get any EOS
        """
        if len(self.finished) == 0:
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=0, score=self.all_scores[-1][0], tokenid=self.outputs[-1][0])
            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Create pydot graph representation of the beam.

        :param outputs:
            self.outputs from the beam
        :param dictionary:
            tok 2 word dict to save words in the tree nodes
        :returns:
            pydot graph
        """
        try:
            import pydot
        except ImportError:
            print('Please install pydot package to dump beam visualization')
        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(hyptail))
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0
                node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid, score=all_scores[tstep][hypid], tokenid=token)
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:
                            color = n_best_colors[i]
                        rank = i
                        break
                label = '<{}'.format(dictionary.vec2txt([token]) if dictionary is not None else token) + ' : ' + '{:.{prec}f}>'.format(all_scores[tstep][hypid], prec=3)
                graph.add_node(pydot.Node(node_tail.__repr__(), label=label, fillcolor=color, style='filled', xlabel='{}'.format(rank) if rank is not None else ''))
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node('"{}"'.format(self.HypothesisTail(timestep=revtstep, hypid=prev_id, score=all_scores[revtstep][prev_id], tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
                to_node = graph.get_node('"{}"'.format(self.HypothesisTail(timestep=revtstep + 1, hypid=i, score=all_scores[revtstep + 1][i], tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)
        return graph
from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
class EndToEndAgent(_GenericWizardAgent):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2 ** 20))(self._vectorize_text)
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']

    def _dummy_batch(self, bsz, maxlen):
        batch = super()._dummy_batch(bsz, maxlen)
        batch['know_vec'] = th.zeros(bsz, 2, 2).long().cuda()
        ck_mask = (th.ones(bsz, 2, dtype=th.uint8) != 0).cuda()
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = th.zeros(bsz).long().cuda()
        batch['use_cs_ids'] = True
        return batch

    def compute_loss(self, batch, return_output=False):
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_tokens = notnull.long().sum().item()
        encoder_states = model_output[2]
        ctx_know_attn = encoder_states[2]
        if self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc
            know_loss = th.nn.functional.cross_entropy(ctx_know_attn, batch.cs_ids, reduction='mean')
            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            know_loss /= num_tokens
            loss = (1 - self.knowledge_alpha) * token_loss + self.knowledge_alpha * know_loss
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['bsz'] = 0.0
        self.metrics['know_acc'] = 0.0
        self.metrics['know_loss'] = 0.0
        self.metrics['know_chance'] = 0.0

    def report(self):
        r = super().report()
        bsz = max(self.metrics['bsz'], 1)
        for k in ['know_loss', 'know_acc', 'know_chance']:
            r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
        return r

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            return list(obs['knowledge_parsed'])
        if 'checked_sentence' not in obs:
            obs_know = [k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            return obs['knowledge_parsed']
        checked_sentence = '{} {} {}'.format(obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence'])
        obs_know = [k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')]
        obs_know = [k for k in obs_know if k]
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = (obs_know[i], obs_know[0])
        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        return obs['knowledge_parsed']

    def batchify(self, obs_batch):
        """
        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]
        all_knowledges = []
        knowledge_counts = []
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            if is_training and self.max_knowledge and (len(obs_know) > self.max_knowledge):
                keepers = 1 + np.random.choice(len(obs_know) - 1, self.max_knowledge, False)
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))
        N = len(reordered_observations)
        K = max(knowledge_counts)
        for i in range(N):
            all_knowledges[i] += [''] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))
        knowledge_vec = [self._vectorize_text(k, truncate=self.knowledge_truncate, add_end=True, truncate_left=False) for k in flattened_knowledge]
        knowledge_vec, _ = padded_tensor(knowledge_vec, self.NULL_IDX, self.use_cuda, left_padded=True)
        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)
        bsz = len(reordered_observations)
        ck_mask = th.zeros(bsz, K, dtype=th.uint8)
        for i, klen in enumerate(knowledge_counts):
            ck_mask[i, :klen] = 1
        ck_mask = ck_mask != 0
        cs_ids = th.LongTensor(bsz).zero_()
        if self.use_cuda:
            knowledge_vec = knowledge_vec.cuda()
            ck_mask = ck_mask.cuda()
            cs_ids = cs_ids.cuda()
        batch['know_vec'] = knowledge_vec
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = cs_ids
        batch['use_cs_ids'] = is_training
        batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
        return batch

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(EndToEndAgent, cls).add_cmdline_args(argparser)
        group = argparser.add_argument_group('EndToEnd Agent')
        group.add_argument('--knowledge-alpha', type=float, default=0.95, help='Weight on the knowledge-attn loss')
        group.add_argument('--knowledge-truncate', type=int, default=32, help='Knowledge truncation field. Defaults to same as --truncate.')
        group.add_argument('--max-knowledge', type=int, help='Reduce the amount of negative knowledge at train time.')
        argparser.add_argument('--knowledge-alpha', type=float, default=0.95, help='Weight on the knowledge-attn loss')

    def _model_input(self, batch):
        return (batch.text_vec, batch.know_vec, batch.ck_mask, batch.cs_ids, batch.use_cs_ids)

    def build_model(self):
        self.model = EndToEndModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(self.model.embeddings.weight, self.opt['embedding_type'])
        if self.use_cuda:
            self.model = self.model.cuda()
        return self.model
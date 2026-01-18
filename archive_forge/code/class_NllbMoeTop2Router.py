import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_nllb_moe import NllbMoeConfig
class NllbMoeTop2Router(nn.Module):
    """
    Router using tokens choose top-2 experts assignment.

    This router uses the same mechanism as in NLLB-MoE from the fairseq repository. Items are sorted by router_probs
    and then routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee
    that each token is processed by an expert**, or that each expert receives at least one token.

    The router combining weights are also returned to make sure that the states that are not updated will be masked.

    """

    def __init__(self, config: NllbMoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.router_ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)
        self.second_expert_policy = config.second_expert_policy
        self.normalize_router_prob_before_dropping = config.normalize_router_prob_before_dropping
        self.batch_prioritized_routing = config.batch_prioritized_routing
        self.moe_eval_capacity_token_fraction = config.moe_eval_capacity_token_fraction

    def _cast_classifier(self):
        """
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, 'SCB') or hasattr(self.classifier, 'CB')):
            self.classifier = self.classifier.to(self.dtype)

    def normalize_router_probabilities(self, router_probs, top_1_mask, top_2_mask):
        top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
        top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
        denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
        top_1_max_probs = top_1_max_probs / denom_s
        top_2_max_probs = top_2_max_probs / denom_s
        return (top_1_max_probs, top_2_max_probs)

    def route_tokens(self, router_logits: torch.Tensor, input_dtype: torch.dtype=torch.float32, padding_mask: Optional[torch.LongTensor]=None) -> Tuple:
        """
        Computes the `dispatch_mask` and the `dispatch_weights` for each experts. The masks are adapted to the expert
        capacity.
        """
        nb_tokens = router_logits.shape[0]
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(input_dtype)
        top_1_expert_index = torch.argmax(router_probs, dim=-1)
        top_1_mask = torch.nn.functional.one_hot(top_1_expert_index, num_classes=self.num_experts)
        if self.second_expert_policy == 'sampling':
            gumbel = torch.distributions.gumbel.Gumbel(0, 1).rsample
            router_logits += gumbel(router_logits.shape).to(router_logits.device)
        logits_except_top_1 = router_logits.masked_fill(top_1_mask.bool(), float('-inf'))
        top_2_expert_index = torch.argmax(logits_except_top_1, dim=-1)
        top_2_mask = torch.nn.functional.one_hot(top_2_expert_index, num_classes=self.num_experts)
        if self.normalize_router_prob_before_dropping:
            top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(router_probs, top_1_mask, top_2_mask)
        if self.second_expert_policy == 'random':
            top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
            sampled = 2 * top_2_max_probs > torch.rand_like(top_2_max_probs.float())
            top_2_mask = top_2_mask * sampled.repeat(self.num_experts, 1).transpose(1, 0)
        if padding_mask is not None and (not self.router_ignore_padding_tokens):
            if len(padding_mask.shape) == 4:
                padding_mask = padding_mask[:, :, -1, :].reshape(-1)[-nb_tokens:]
            non_padding = ~padding_mask.bool()
            top_1_mask = top_1_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)
            top_2_mask = top_2_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)
        if self.batch_prioritized_routing:
            importance_scores = -1 * router_probs.max(dim=1)[0]
            sorted_top_1_mask = top_1_mask[importance_scores.argsort(dim=0)]
            sorted_cumsum1 = (torch.cumsum(sorted_top_1_mask, dim=0) - 1) * sorted_top_1_mask
            locations1 = sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]
            sorted_top_2_mask = top_2_mask[importance_scores.argsort(dim=0)]
            sorted_cumsum2 = (torch.cumsum(sorted_top_2_mask, dim=0) - 1) * sorted_top_2_mask
            locations2 = sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]
            locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)
        else:
            locations1 = torch.cumsum(top_1_mask, dim=0) - 1
            locations2 = torch.cumsum(top_2_mask, dim=0) - 1
            locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)
        if not self.training and self.moe_eval_capacity_token_fraction > 0:
            self.expert_capacity = math.ceil(self.moe_eval_capacity_token_fraction * nb_tokens)
        else:
            capacity = 2 * math.ceil(nb_tokens / self.num_experts)
            self.expert_capacity = capacity if self.expert_capacity is None else self.expert_capacity
        top_1_mask = top_1_mask * torch.lt(locations1, self.expert_capacity)
        top_2_mask = top_2_mask * torch.lt(locations2, self.expert_capacity)
        if not self.normalize_router_prob_before_dropping:
            top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(router_probs, top_1_mask, top_2_mask)
        gates1 = top_1_max_probs[:, None] * top_1_mask
        gates2 = top_2_max_probs[:, None] * top_2_mask
        router_probs = gates1 + gates2
        return (top_1_mask, router_probs)

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor]=None) -> Tuple:
        """
        The hidden states are reshaped to simplify the computation of the router probabilities (combining weights for
        each experts.)

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            top_1_mask (`torch.Tensor` of shape (batch_size, sequence_length)):
                Index tensor of shape [batch_size, sequence_length] corresponding to the expert selected for each token
                using the top1 probabilities of the router.
            router_probabilities (`torch.Tensor` of shape (batch_size, sequence_length, nump_experts)):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor` of shape (batch_size, sequence_length))):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        self.input_dtype = hidden_states.dtype
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * sequence_length, hidden_dim)
        hidden_states = hidden_states.to(self.dtype)
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)
        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        return (top_1_mask, router_probs)
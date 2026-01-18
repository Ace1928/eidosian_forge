import math
from typing import Callable, Optional, Protocol, Tuple
import torch
class BeamSearchSampler:
    """Beam Search sampling algorithm.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self, beams: int=1):
        self.samples = beams

    def __call__(self, next_token_logits: torch.DoubleTensor, sequence_weights: torch.DoubleTensor, _) -> Tuple[torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor]:
        """Call the beam search sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        weights = logprobs + sequence_weights.unsqueeze(1).expand_as(next_token_logits)
        batch_size = next_token_logits.shape[0] // self.samples
        vocab_size = next_token_logits.shape[-1]
        weights = weights.view(batch_size, self.samples * vocab_size)
        if torch.all(sequence_weights == 0):
            weights = weights[:, :vocab_size]
        weights, indices = torch.topk(weights, self.samples, dim=1, largest=True, sorted=True)
        ancestors = torch.div(indices, vocab_size, rounding_mode='floor')
        next_token_ids = indices % vocab_size
        first_batch_idx = torch.arange(0, batch_size * self.samples, self.samples, device=next_token_logits.device).unsqueeze(1)
        ancestors = ancestors + first_batch_idx
        ancestors = ancestors.view(self.samples * batch_size)
        weights = weights.view(self.samples * batch_size)
        next_token_ids = next_token_ids.view(self.samples * batch_size, 1)
        return (next_token_ids, ancestors, weights)
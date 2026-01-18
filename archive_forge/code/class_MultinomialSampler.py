import math
from typing import Callable, Optional, Protocol, Tuple
import torch
class MultinomialSampler:
    """Multinomial sampling algorithm.

    Multinomial sampling consists in randomly sampling the next token assuming
    its distribution is a Categorical distribution parametrized by the
    next-token logits.


    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self, samples: int=1, *, top_k: Optional[int]=None, top_p: Optional[float]=None, temperature: Optional[float]=None):
        self.samples = samples
        self.logits_processors = []
        if top_k is not None:
            self.logits_processors.append(keep_top_k_logits(top_k))
        elif top_p is not None:
            self.logits_processors.append(keep_top_p_logits(top_p))
        if temperature is not None:
            self.logits_processors.append(rescale_logits(temperature))

    def __call__(self, next_token_logits: torch.DoubleTensor, sequence_weights: torch.DoubleTensor, rng: torch.Generator) -> Tuple[torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor]:
        """Call the multinomial sampler.

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
        altered_next_token_logits = next_token_logits
        for logit_processor in self.logits_processors:
            altered_next_token_logits = logit_processor(next_token_logits)
        probs = torch.nn.functional.softmax(altered_next_token_logits, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        logprobs = torch.nn.functional.log_softmax(altered_next_token_logits, dim=-1)
        ancestors = torch.arange(altered_next_token_logits.shape[0], device=next_token_logits.device)
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()
        return (next_token_ids, ancestors, weights)
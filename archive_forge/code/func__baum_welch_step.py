from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _baum_welch_step(self, sequence, model, symbol_to_number):
    N = len(model._states)
    M = len(model._symbols)
    T = len(sequence)
    alpha = model._forward_probability(sequence)
    beta = model._backward_probability(sequence)
    lpk = logsumexp2(alpha[T - 1])
    A_numer = _ninf_array((N, N))
    B_numer = _ninf_array((N, M))
    A_denom = _ninf_array(N)
    B_denom = _ninf_array(N)
    transitions_logprob = model._transitions_matrix().T
    for t in range(T):
        symbol = sequence[t][_TEXT]
        next_symbol = None
        if t < T - 1:
            next_symbol = sequence[t + 1][_TEXT]
        xi = symbol_to_number[symbol]
        next_outputs_logprob = model._outputs_vector(next_symbol)
        alpha_plus_beta = alpha[t] + beta[t]
        if t < T - 1:
            numer_add = transitions_logprob + next_outputs_logprob + beta[t + 1] + alpha[t].reshape(N, 1)
            A_numer = np.logaddexp2(A_numer, numer_add)
            A_denom = np.logaddexp2(A_denom, alpha_plus_beta)
        else:
            B_denom = np.logaddexp2(A_denom, alpha_plus_beta)
        B_numer[:, xi] = np.logaddexp2(B_numer[:, xi], alpha_plus_beta)
    return (lpk, A_numer, A_denom, B_numer, B_denom)
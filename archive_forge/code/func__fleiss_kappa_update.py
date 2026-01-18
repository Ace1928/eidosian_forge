import torch
from torch import Tensor
from typing_extensions import Literal
def _fleiss_kappa_update(ratings: Tensor, mode: Literal['counts', 'probs']='counts') -> Tensor:
    """Updates the counts for fleiss kappa metric.

    Args:
        ratings: ratings matrix
        mode: whether ratings are provided as counts or probabilities

    """
    if mode == 'probs':
        if ratings.ndim != 3 or not ratings.is_floating_point():
            raise ValueError("If argument ``mode`` is 'probs', ratings must have 3 dimensions with the format [n_samples, n_categories, n_raters] and be floating point.")
        ratings = ratings.argmax(dim=1)
        one_hot = torch.nn.functional.one_hot(ratings, num_classes=ratings.shape[1]).permute(0, 2, 1)
        ratings = one_hot.sum(dim=-1)
    elif mode == 'counts' and (ratings.ndim != 2 or ratings.is_floating_point()):
        raise ValueError('If argument ``mode`` is `counts`, ratings must have 2 dimensions with the format [n_samples, n_categories] and be none floating point.')
    return ratings
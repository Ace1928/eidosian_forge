import torch
from torch import Tensor
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
Compute concordance correlation coefficient that measures the agreement between two variables.

    .. math::
        \rho_c = \frac{2 \rho \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}

    where :math:`\mu_x, \mu_y` is the means for the two variables, :math:`\sigma_x^2, \sigma_y^2` are the corresponding
    variances and \rho is the pearson correlation coefficient between the two variables.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from torchmetrics.functional.regression import concordance_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> concordance_corrcoef(preds, target)
        tensor([0.9777])

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import concordance_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> concordance_corrcoef(preds, target)
        tensor([0.7273, 0.9887])

    
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import adaptive_avg_pool2d
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _torch_fidelity_forward(self, x: Tensor) -> Tuple[Tensor, ...]:
    """Forward method of inception net.

        Copy of the forward method from this file:
        https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py
        with a single line change regarding the casting of `x` in the beginning.

        Corresponding license file (Apache License, Version 2.0):
        https://github.com/toshas/torch-fidelity/blob/master/LICENSE.md

        """
    vassert(torch.is_tensor(x) and x.dtype == torch.uint8, 'Expecting image as torch.Tensor with dtype=torch.uint8')
    features = {}
    remaining_features = self.features_list.copy()
    x = x.to(self._dtype) if hasattr(self, '_dtype') else x.to(torch.float)
    x = interpolate_bilinear_2d_like_tensorflow1x(x, size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), align_corners=False)
    x = (x - 128) / 128
    x = self.Conv2d_1a_3x3(x)
    x = self.Conv2d_2a_3x3(x)
    x = self.Conv2d_2b_3x3(x)
    x = self.MaxPool_1(x)
    if '64' in remaining_features:
        features['64'] = adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        remaining_features.remove('64')
        if len(remaining_features) == 0:
            return tuple((features[a] for a in self.features_list))
    x = self.Conv2d_3b_1x1(x)
    x = self.Conv2d_4a_3x3(x)
    x = self.MaxPool_2(x)
    if '192' in remaining_features:
        features['192'] = adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        remaining_features.remove('192')
        if len(remaining_features) == 0:
            return tuple((features[a] for a in self.features_list))
    x = self.Mixed_5b(x)
    x = self.Mixed_5c(x)
    x = self.Mixed_5d(x)
    x = self.Mixed_6a(x)
    x = self.Mixed_6b(x)
    x = self.Mixed_6c(x)
    x = self.Mixed_6d(x)
    x = self.Mixed_6e(x)
    if '768' in remaining_features:
        features['768'] = adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        remaining_features.remove('768')
        if len(remaining_features) == 0:
            return tuple((features[a] for a in self.features_list))
    x = self.Mixed_7a(x)
    x = self.Mixed_7b(x)
    x = self.Mixed_7c(x)
    x = self.AvgPool(x)
    x = torch.flatten(x, 1)
    if '2048' in remaining_features:
        features['2048'] = x
        remaining_features.remove('2048')
        if len(remaining_features) == 0:
            return tuple((features[a] for a in self.features_list))
    if 'logits_unbiased' in remaining_features:
        x = x.mm(self.fc.weight.T)
        features['logits_unbiased'] = x
        remaining_features.remove('logits_unbiased')
        if len(remaining_features) == 0:
            return tuple((features[a] for a in self.features_list))
        x = x + self.fc.bias.unsqueeze(0)
    else:
        x = self.fc(x)
    features['logits'] = x
    return tuple((features[a] for a in self.features_list))
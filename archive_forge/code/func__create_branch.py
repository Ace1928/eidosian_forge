import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def _create_branch(d_model: int, nhead: int, metric: str) -> nn.modules.Module:
    """Create branch module after DPRNN model for predicting metric score.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): Number of heads in the multi-head attention model.
        metric (str): The metric name to predict.

    Returns:
        (nn.Module): Returned module to predict corresponding metric score.
    """
    layer1 = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.0, batch_first=True)
    layer2 = AutoPool()
    if metric == 'stoi':
        layer3 = nn.Sequential(nn.Linear(d_model, d_model), nn.PReLU(), nn.Linear(d_model, 1), RangeSigmoid())
    elif metric == 'pesq':
        layer3 = nn.Sequential(nn.Linear(d_model, d_model), nn.PReLU(), nn.Linear(d_model, 1), RangeSigmoid(val_range=PESQRange))
    else:
        layer3: nn.modules.Module = nn.Sequential(nn.Linear(d_model, d_model), nn.PReLU(), nn.Linear(d_model, 1))
    return nn.Sequential(layer1, layer2, layer3)
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional
import torch
def _decrease_loss_scale(self) -> None:
    self.loss_scale /= self.scale_factor
    if self.threshold is not None:
        self.loss_scale = max(self.loss_scale, self.threshold)
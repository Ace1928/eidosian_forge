import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from xformers._deprecation_warning import deprecated_function
from xformers.components import reversible as rv
from xformers.components.residual import ResidualNormStyle, get_deepnorm_coefficients
from xformers.factory.block_configs import (
from xformers.factory.block_factory import xFormerDecoderBlock, xFormerEncoderBlock
from xformers.factory.weight_init import get_weight_init_fn, xFormerWeightInit
def _verify_deepnorm(self, stack_configs: List[xFormerBlockConfig]):
    deepnorm = [c.residual_norm_style == ResidualNormStyle.DeepNorm for c in stack_configs]
    assert all(deepnorm) or not any(deepnorm), 'All layers need to have the same deepnorm setting. ' + f'Currently {deepnorm}'
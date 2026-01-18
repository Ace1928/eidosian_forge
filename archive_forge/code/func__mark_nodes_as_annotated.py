import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if QUANT_ANNOTATION_KEY not in node.meta:
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation()
            node.meta[QUANT_ANNOTATION_KEY]._annotated = True
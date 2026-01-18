from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _nodes_are_equal(self, pn: Node, gn: Node) -> bool:
    if not self.match_placeholder and pn.op == 'placeholder':
        return True
    if pn.op == gn.op:
        if pn.op == 'placeholder' or pn.op == 'output':
            return True
        elif pn.op == 'get_attr':
            return self._match_attributes(pn, gn)
        return pn.target == gn.target
    return False
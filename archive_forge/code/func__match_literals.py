from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _match_literals(self, pn: Any, gn: Any, match: InternalMatch) -> bool:
    assert not (isinstance(pn, Node) and isinstance(gn, Node)), 'pn and gn cannot both be Node'
    if isinstance(pn, Node) and (not isinstance(gn, Node)):
        if pn.op == 'placeholder':
            if pn in match.nodes_map:
                return match.nodes_map[pn] == gn
            match.nodes_map[pn] = gn
            return True
        else:
            return False
    elif not isinstance(pn, Node) and isinstance(gn, Node):
        return False
    else:
        return type(gn) == type(pn) and gn == pn
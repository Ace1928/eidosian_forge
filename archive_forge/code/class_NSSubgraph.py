import enum
from typing import NamedTuple
from torch.fx.graph import Node
from typing import Dict, Any, List, Union, Callable
class NSSubgraph(NamedTuple):
    start_node: Node
    end_node: Node
    base_op_node: Node
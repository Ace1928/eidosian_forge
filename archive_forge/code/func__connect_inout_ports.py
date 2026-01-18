import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def _connect_inout_ports(self, last_node, node, builder):
    if isinstance(last_node, DDGProtocol) and isinstance(node, DDGProtocol):
        for k in last_node.outgoing_states:
            builder.graph.add_edge(f'outgoing_{last_node.name}', f'incoming_{node.name}', src_port=k, dst_port=k)
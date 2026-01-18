import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module

    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.
    
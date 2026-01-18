from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Dict, List, Any, Type, Optional, Callable
import logging
import os
def _init_logger():
    logger = logging.getLogger(__name__)
    level = os.environ.get('PYTORCH_MATCHER_LOGLEVEL', 'WARNING').upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(filename)s > %(message)s')
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger
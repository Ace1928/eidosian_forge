from typing import NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from gym.logger import warn
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.space import Space
class GraphInstance(NamedTuple):
    """A Graph space instance.

    * nodes (np.ndarray): an (n x ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
    * edges (Optional[np.ndarray]): an (m x ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
    * edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
    """
    nodes: np.ndarray
    edges: Optional[np.ndarray]
    edge_links: Optional[np.ndarray]
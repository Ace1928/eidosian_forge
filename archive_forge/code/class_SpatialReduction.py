from typing import Optional
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils
class SpatialReduction(Reduction):
    """Reduce the dimension of a spatial tensor, e.g. image, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type: Optional[str]=None, **kwargs):
        super().__init__(reduction_type, **kwargs)

    def global_max(self, input_node):
        return layer_utils.get_global_max_pooling(input_node.shape)()(input_node)

    def global_avg(self, input_node):
        return layer_utils.get_global_average_pooling(input_node.shape)()(input_node)
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import _pywrap_tf_item as tf_item
def _BuildTFItem(self):
    self._tf_item = tf_item.TF_NewItem(self._metagraph.SerializeToString(), self._ignore_colocation, self._ignore_user_placement)
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import _pywrap_tf_item as tf_item
def GetOpProperties(self):
    """Get Op properties."""
    props = tf_item.TF_GetOpProperties(self.tf_item)
    properties = {}
    for key, values in props.items():
        prop = []
        for value in values:
            prop.append(op_performance_data_pb2.OpInfo.TensorProperties.FromString(value))
        properties[key] = prop
    return properties
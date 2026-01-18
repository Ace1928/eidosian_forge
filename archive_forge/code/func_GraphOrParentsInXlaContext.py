from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables
def GraphOrParentsInXlaContext(graph):
    while True:
        if InXlaContext(graph):
            return True
        try:
            graph = graph.outer_graph
        except AttributeError:
            return False
from tensorboard.compat.proto import graph_pb2
def _prefixed_op_name(prefix, op_name):
    return '%s/%s' % (prefix, op_name)
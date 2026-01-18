from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
@ops.RegisterGradient('CudnnRNN')
def _cudnn_rnn_backward(op, *grads):
    """Gradients for the CudnnRNN op."""
    if not op.get_attr('is_training'):
        raise ValueError('To use CudnnRNN in gradients, is_training must be set to True.')
    return gen_cudnn_rnn_ops.cudnn_rnn_backprop(input=op.inputs[0], input_h=op.inputs[1], input_c=op.inputs[2], params=op.inputs[3], output=op.outputs[0], output_h=op.outputs[1], output_c=op.outputs[2], output_backprop=grads[0], output_h_backprop=grads[1], output_c_backprop=grads[2], reserve_space=op.outputs[3], dropout=op.get_attr('dropout'), seed=op.get_attr('seed'), seed2=op.get_attr('seed2'), rnn_mode=op.get_attr('rnn_mode'), input_mode=op.get_attr('input_mode'), direction=op.get_attr('direction'))
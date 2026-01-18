from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _fix_gemm(op_name, inputs, old_attr, proto_obj):
    """Using FullyConnected operator in place of linalg_gemm to perform same operation"""
    op_sym = getattr(symbol, op_name, None)
    alpha = float(old_attr.get('alpha', 1.0))
    beta = float(old_attr.get('beta', 1.0))
    trans_a = int(old_attr.get('transA', 0))
    trans_b = int(old_attr.get('transB', 0))
    if trans_a:
        inputs[0] = symbol.transpose(inputs[0], axes=(1, 0))
    if not trans_b:
        inputs[1] = symbol.transpose(inputs[1], axes=(1, 0))
    new_inputs = [alpha * inputs[0], inputs[1], beta * inputs[2]]
    new_attr = {'num_hidden': proto_obj._params[inputs[2].name].shape[0]}
    return (op_sym, new_attr, new_inputs)
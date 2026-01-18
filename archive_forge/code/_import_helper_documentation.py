from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
from ._op_translations import add, subtract, multiply, divide, absolute, negative, add_n
from ._op_translations import tanh, arccos, arcsin, arctan, _cos, _sin, _tan
from ._op_translations import softplus, shape, gather, lp_pooling, size
from ._op_translations import ceil, floor, hardsigmoid, global_lppooling
from ._op_translations import concat, hardmax, topk
from ._op_translations import leaky_relu, _elu, _prelu, _selu, softmax, fully_connected
from ._op_translations import global_avgpooling, global_maxpooling, linalg_gemm
from ._op_translations import sigmoid, pad, relu, matrix_multiplication, batch_norm
from ._op_translations import dropout, local_response_norm, conv, deconv
from ._op_translations import reshape, cast, split, _slice, transpose, squeeze, flatten
from ._op_translations import reciprocal, squareroot, power, exponent, _log, unsqueeze
from ._op_translations import reduce_max, reduce_mean, reduce_min, reduce_sum
from ._op_translations import reduce_prod, avg_pooling, max_pooling, instance_norm
from ._op_translations import argmax, argmin, maximum, minimum
from ._op_translations import clip, reduce_log_sum, reduce_log_sum_exp
from ._op_translations import reduce_sum_square, reduce_l1, reduce_l2, max_roi_pooling
from ._op_translations import log_softmax, softsign, lesser, greater, equal
from ._op_translations import logical_and, logical_or, logical_xor, logical_not
from ._op_translations import mean, depthtospace, spacetodepth, lpnormalization
Operator attributes conversion
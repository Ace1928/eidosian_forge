from ._internal import NDArrayBase
from ..base import _Null
def _sg_mkldnn_selfatt_valatt(queries_keys_values=None, attention=None, heads=_Null, quantized=_Null, enable_float_output=_Null, min_calib_range=_Null, max_calib_range=_Null, out=None, name=None, **kwargs):
    """_sg_mkldnn_selfatt_valatt

    Defined in ../src/operator/subgraph/mkldnn/mkldnn_transformer.cc:L612

    Parameters
    ----------
    queries_keys_values : NDArray
        Queries, keys and values interleaved
    attention : NDArray
        Attention maps
    heads : int, required
        Set number of heads
    quantized : boolean, optional, default=0
        Whether it's a quantized InterleavedMatMul operator
    enable_float_output : boolean, optional, default=0
        Whether to enable float32 output
    min_calib_range : float or None, optional, default=None
        The minimum scalar value in the form of float32 obtained through calibration. If present, it will be used to by quantized InterleavedMatMul op to calculate primitive scale
    max_calib_range : float or None, optional, default=None
        The maximum scalar value in the form of float32 obtained through calibration. If present, it will be used to by quantized InterleavedMatMul op to calculate primitive scale

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)
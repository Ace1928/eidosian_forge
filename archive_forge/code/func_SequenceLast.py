from ._internal import NDArrayBase
from ..base import _Null
def SequenceLast(data=None, sequence_length=None, use_sequence_length=_Null, axis=_Null, out=None, name=None, **kwargs):
    """Takes the last element of a sequence.

    This function takes an n-dimensional input array of the form
    [max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array
    of the form [batch_size, other_feature_dims].

    Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be
    an input array of positive ints of dimension [batch_size]. To use this parameter,
    set `use_sequence_length` to `True`, otherwise each example in the batch is assumed
    to have the max sequence length.

    .. note:: Alternatively, you can also use `take` operator.

    Example::

       x = [[[  1.,   2.,   3.],
             [  4.,   5.,   6.],
             [  7.,   8.,   9.]],

            [[ 10.,   11.,   12.],
             [ 13.,   14.,   15.],
             [ 16.,   17.,   18.]],

            [[  19.,   20.,   21.],
             [  22.,   23.,   24.],
             [  25.,   26.,   27.]]]

       // returns last sequence when sequence_length parameter is not used
       SequenceLast(x) = [[  19.,   20.,   21.],
                          [  22.,   23.,   24.],
                          [  25.,   26.,   27.]]

       // sequence_length is used
       SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
                [[  1.,   2.,   3.],
                 [  4.,   5.,   6.],
                 [  7.,   8.,   9.]]

       // sequence_length is used
       SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
                [[  1.,    2.,   3.],
                 [  13.,  14.,  15.],
                 [  25.,  26.,  27.]]



    Defined in ../src/operator/sequence_last.cc:L105

    Parameters
    ----------
    data : NDArray
        n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2
    sequence_length : NDArray
        vector of sequence lengths of the form [batch_size]
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
    axis : int, optional, default='0'
        The sequence axis. Only values of 0 and 1 are currently supported.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)
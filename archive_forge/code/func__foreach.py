from ._internal import NDArrayBase
from ..base import _Null
def _foreach(*data, **kwargs):
    """Run a for loop over an NDArray with user-defined computation

    From:../src/operator/control_flow.cc:1090

    Parameters
    ----------
    fn : Symbol
        Input graph.
    data : NDArray[]
        The input arrays that include data arrays and states.
    num_outputs : int, required
        The number of outputs of the subgraph.
    num_out_data : int, required
        The number of output data of the subgraph.
    in_state_locs : tuple of <long>, required
        The locations of loop states among the inputs.
    in_data_locs : tuple of <long>, required
        The locations of input data among the inputs.
    remain_locs : tuple of <long>, required
        The locations of remaining data among the inputs.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)
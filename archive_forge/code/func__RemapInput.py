import pickle
import numpy
from rdkit.ML.Data import DataUtils
def _RemapInput(self, inputVect):
    """ remaps the input so that it matches the expected internal ordering

      **Arguments**

        - inputVect: the input to be reordered

      **Returns**

        - a list with the reordered (and possible shorter) data

      **Note**

        - you must call _SetDescriptorNames()_ and _SetInputOrder()_ for this to work

        - this is primarily intended for internal use

    """
    order = self._mapOrder
    if order is None:
        return inputVect
    remappedInput = [None] * len(order)
    for i in range(len(order) - 1):
        remappedInput[i] = inputVect[order[i]]
    if order[-1] == -1:
        remappedInput[-1] = 0
    else:
        remappedInput[-1] = inputVect[order[-1]]
    return remappedInput
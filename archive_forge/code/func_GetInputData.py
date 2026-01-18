import copy
import math
import numpy
def GetInputData(self):
    """ returns the input data

         **Note**

           _inputData_ means the examples without their result fields
            (the last _NResults_ entries)

        """
    return self.data[:, :-self.nResults].tolist()
import numpy
from rdkit.ML.InfoTheory import entropy
 finds multiple quantization bounds for a single variable

     **Arguments**

       - vals: sequence of variable values (assumed to be floats)

       - nBounds: the number of quantization bounds to find

       - results: a list of result codes (should be integers)

       - nPossibleRes: an integer with the number of possible values of the
         result variable

     **Returns**

       - a 2-tuple containing:

         1) a list of the quantization bounds (floats)

         2) the information gain associated with this quantization


    
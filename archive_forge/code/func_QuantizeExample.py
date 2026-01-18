import pickle
import numpy
from rdkit.ML.Data import DataUtils
def QuantizeExample(self, example, quantBounds=None):
    """ quantizes an example

      **Arguments**

       - example: a data point (list, tuple or numpy array)

       - quantBounds:  a list of quantization bounds, each quantbound is a
             list of boundaries.  If this argument is not provided, the composite
             will use its own quantBounds

      **Returns**

        the quantized example as a list

      **Notes**

        - If _example_ is different in length from _quantBounds_, this will
           assert out.

        - This is primarily intended for internal use

    """
    if quantBounds is None:
        quantBounds = self.quantBounds
    assert len(example) == len(quantBounds), 'example/quantBounds mismatch'
    quantExample = [None] * len(example)
    for i in range(len(quantBounds)):
        bounds = quantBounds[i]
        p = example[i]
        if len(bounds):
            for box in range(len(bounds)):
                if p < bounds[box]:
                    p = box
                    break
            else:
                p = box + 1
        elif i != 0:
            p = int(p)
        quantExample[i] = p
    return quantExample
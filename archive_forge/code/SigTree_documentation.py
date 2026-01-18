import copy
from rdkit.DataStructs.VectCollection import VectCollection
from rdkit.ML.DecTree import DecTree
 Recursively classify an example by running it through the tree

      **Arguments**

        - example: the example to be classified, a sequence at least
          2 long:
           ( id, sig )
          where sig is a BitVector (or something supporting __getitem__)
          additional fields will be ignored.

        - appendExamples: if this is nonzero then this node (and all children)
          will store the example

      **Returns**

        the classification of _example_

    
import numpy
from rdkit.ML.Composite import Composite
def CompositeToBayesComposite(obj):
    """ converts a Composite to a BayesComposite

   if _obj_ is already a BayesComposite or if it is not a _Composite.Composite_ ,
    nothing will be done.

  """
    if obj.__class__ == BayesComposite:
        return
    elif obj.__class__ == Composite.Composite:
        obj.__class__ = BayesComposite
        obj.resultProbs = None
        obj.condProbs = None
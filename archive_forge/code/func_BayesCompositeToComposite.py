import numpy
from rdkit.ML.Composite import Composite
def BayesCompositeToComposite(obj):
    """ converts a BayesComposite to a Composite.Composite

  """
    if obj.__class__ == Composite.Composite:
        return
    elif obj.__class__ == BayesComposite:
        obj.__class__ = Composite.Composite
        obj.resultProbs = None
        obj.condProbs = None
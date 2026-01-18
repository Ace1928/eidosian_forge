from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def CalcDescriptorsForComposition(self, composVect, propDict):
    """ calculates all descriptors for a given composition

      **Arguments**

        - compos: a string representation of the composition

        - propDict: a dictionary containing the properties of the composition
          as a whole (e.g. structural variables, etc.). These are used to
          generate Compound Descriptors

      **Returns**
        the list of all descriptor values

      **Notes**

        - this uses _chemutils.SplitComposition_
          to split the composition into its individual pieces

    """
    composList = chemutils.SplitComposition(composVect[0])
    try:
        r1 = self.CalcSimpleDescriptorsForComposition(composList=composList)
    except KeyError:
        res = []
    else:
        r2 = self.CalcCompoundDescriptorsForComposition(composList=composList, propDict=propDict)
        res = r1 + r2
    return tuple(res)
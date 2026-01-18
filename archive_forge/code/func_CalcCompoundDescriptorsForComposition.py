from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def CalcCompoundDescriptorsForComposition(self, compos='', composList=None, propDict={}):
    """ calculates all simple descriptors for a given composition

      **Arguments**

        - compos: a string representation of the composition

        - composList: a *composVect*

        - propDict: a dictionary containing the properties of the composition
          as a whole (e.g. structural variables, etc.)

        The client must provide either _compos_ or _composList_.  If both are
        provided, _composList_ takes priority.

      **Returns**
        the list of descriptor values

      **Notes**

        - when _compos_ is provided, this uses _chemutils.SplitComposition_
          to split the composition into its individual pieces

    """
    if composList is None:
        composList = chemutils.SplitComposition(compos)
    res = []
    for cl in self.compoundList:
        val = Parser.CalcSingleCompoundDescriptor(composList, cl[1:], self.atomDict, propDict)
        res.append(val)
    return res
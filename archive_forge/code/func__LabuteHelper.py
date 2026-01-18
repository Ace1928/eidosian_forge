import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
def _LabuteHelper(mol, includeHs=1, force=0):
    """ *Internal Use Only*
    helper function for LabuteASA calculation
    returns an array of atomic contributions to the ASA

  **Note:** Changes here affect the version numbers of all ASA descriptors

  """
    if not force:
        try:
            res = mol._labuteContribs
        except AttributeError:
            pass
        else:
            if res:
                return res
    tpl = rdMolDescriptors._CalcLabuteASAContribs(mol, includeHs)
    ats, hs = tpl
    Vi = [hs] + list(ats)
    mol._labuteContribs = Vi
    return Vi
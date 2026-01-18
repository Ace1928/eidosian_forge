import bisect
import numpy
from rdkit.Chem.EState.EState import EStateIndices as EStateIndices_
from rdkit.Chem.MolSurf import _LabuteHelper as VSAContribs_
def _descriptor_VSA_EState(nbin):

    def VSA_EState_bin(mol):
        return VSA_EState_(mol, force=False)[nbin]
    name = 'VSA_EState{0}'.format(nbin + 1)
    fn = VSA_EState_bin
    fn.__doc__ = _descriptorDocstring('VSA EState', nbin, vsaBins)
    fn.version = '1.0.0'
    return (name, fn)
import bisect
import numpy
from rdkit.Chem.EState.EState import EStateIndices as EStateIndices_
from rdkit.Chem.MolSurf import _LabuteHelper as VSAContribs_
def _descriptor_EState_VSA(nbin):

    def EState_VSA_bin(mol):
        return EState_VSA_(mol, force=False)[nbin]
    name = 'EState_VSA{0}'.format(nbin + 1)
    fn = EState_VSA_bin
    fn.__name__ = name
    if hasattr(fn, '__qualname__'):
        fn.__qualname__ = name
    fn.__doc__ = _descriptorDocstring('EState VSA', nbin, estateBins)
    fn.version = '1.0.1'
    return (name, fn)
import copy
import re
import sys
from rdkit import Chem
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
 returns the BRICS decomposition for a molecule

    >>> from rdkit import Chem
    >>> m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
    >>> res = list(BRICSDecompose(m))
    >>> sorted(res)
    ['[14*]c1ccccn1', '[16*]c1cccc([16*])c1', '[3*]O[3*]', '[4*]CCC', '[4*]C[8*]']

    >>> res = list(BRICSDecompose(m,returnMols=True))
    >>> res[0]
    <rdkit.Chem.rdchem.Mol object ...>
    >>> smis = [Chem.MolToSmiles(x,True) for x in res]
    >>> sorted(smis)
    ['[14*]c1ccccn1', '[16*]c1cccc([16*])c1', '[3*]O[3*]', '[4*]CCC', '[4*]C[8*]']

    nexavar, an example from the paper (corrected):

    >>> m = Chem.MolFromSmiles('CNC(=O)C1=NC=CC(OC2=CC=C(NC(=O)NC3=CC(=C(Cl)C=C3)C(F)(F)F)C=C2)=C1')
    >>> res = list(BRICSDecompose(m))
    >>> sorted(res)
    ['[1*]C([1*])=O', '[1*]C([6*])=O', '[14*]c1cc([16*])ccn1', '[16*]c1ccc(Cl)c([16*])c1', '[16*]c1ccc([16*])cc1', '[3*]O[3*]', '[5*]NC', '[5*]N[5*]', '[8*]C(F)(F)F']

    it's also possible to keep pieces that haven't been fully decomposed:

    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True))
    >>> sorted(res)
    ['CCCOCC', '[3*]OCC', '[3*]OCCC', '[3*]O[3*]', '[4*]CC', '[4*]CCC']

    >>> m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True))
    >>> sorted(res)
    ['CCCOCc1cccc(-c2ccccn2)c1', '[14*]c1ccccn1', '[16*]c1cccc(-c2ccccn2)c1', '[16*]c1cccc(COCCC)c1', '[16*]c1cccc([16*])c1', '[3*]OCCC', '[3*]OC[8*]', '[3*]OCc1cccc(-c2ccccn2)c1', '[3*]OCc1cccc([16*])c1', '[3*]O[3*]', '[4*]CCC', '[4*]C[8*]', '[4*]Cc1cccc(-c2ccccn2)c1', '[4*]Cc1cccc([16*])c1', '[8*]COCCC']

    or to only do a single pass of decomposition:

    >>> m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
    >>> res = list(BRICSDecompose(m,singlePass=True))
    >>> sorted(res)
    ['CCCOCc1cccc(-c2ccccn2)c1', '[14*]c1ccccn1', '[16*]c1cccc(-c2ccccn2)c1', '[16*]c1cccc(COCCC)c1', '[3*]OCCC', '[3*]OCc1cccc(-c2ccccn2)c1', '[4*]CCC', '[4*]Cc1cccc(-c2ccccn2)c1', '[8*]COCCC']

    setting a minimum size for the fragments:

    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True,minFragmentSize=2))
    >>> sorted(res)
    ['CCCOCC', '[3*]OCC', '[3*]OCCC', '[4*]CC', '[4*]CCC']
    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True,minFragmentSize=3))
    >>> sorted(res)
    ['CCCOCC', '[3*]OCC', '[4*]CCC']
    >>> res = list(BRICSDecompose(m,minFragmentSize=2))
    >>> sorted(res)
    ['[3*]OCC', '[3*]OCCC', '[4*]CC', '[4*]CCC']


    
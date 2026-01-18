import os.path
import re
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
def _readPattyDefs(fname=os.path.join(RDConfig.RDDataDir, 'SmartsLib', 'patty_rules.txt')):
    with open(fname, 'r') as inf:
        lines = [x.strip().split('# ')[0].strip() for x in inf]
    splitl = [re.split('[ ]+', x) for x in lines if x != '']
    matchers = []
    for tpl in splitl:
        if len(tpl) > 1:
            mol = Chem.MolFromSmarts(tpl[0])
            if mol is None:
                continue
            matchers.append((mol, tpl[1]))
    return matchers
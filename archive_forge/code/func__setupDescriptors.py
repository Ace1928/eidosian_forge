from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import _isCallable
def _setupDescriptors(namespace):
    global descList
    descList.clear()
    for nm, thing in tuple(namespace.items()):
        if nm[0] != '_' and nm != 'CalcMolDescriptors3D' and _isCallable(thing):
            descList.append((nm, thing))
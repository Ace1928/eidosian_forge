import collections
from rdkit import Chem
from rdkit.Chem import AllChem

    Enumerate possible relevant heterocycles on the given input
    molecule.

    >>> from rdkit.Chem.EnumerateHeterocycles import EnumerateHeterocycles
    >>> from rdkit import Chem
    >>> for smi in sorted(Chem.MolToSmiles(m) for m in EnumerateHeterocycles(Chem.MolFromSmiles('c1ccccc1'))):
    ...     print(smi)
    c1ccccc1
    c1ccncc1
    c1ccnnc1
    c1cnccn1
    c1cncnc1
    c1cnncn1
    c1cnnnc1
    c1ncncn1

    The algorithm works by mutating only one atom at a time. The depth
    parameter can be used to control the level of this recursion. For
    example, only enumerating aromatic rings that are one atom different:

    >>> for smi in sorted(Chem.MolToSmiles(m) for m in EnumerateHeterocycles(Chem.MolFromSmiles('n1ccccc1'), depth=1)):
    ...     print(smi)
    c1ccccc1
    c1ccnnc1
    c1cnccn1
    c1cncnc1
    
import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
def EnumerateStereoisomers(m, options=StereoEnumerationOptions(), verbose=False):
    """ returns a generator that yields possible stereoisomers for a molecule

    Arguments:
      - m: the molecule to work with
      - options: parameters controlling the enumeration
      - verbose: toggles how verbose the output is

    If m has stereogroups, they will be expanded

    A small example with 3 chiral atoms and 1 chiral bond (16 theoretical stereoisomers):

    >>> from rdkit import Chem
    >>> from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    >>> m = Chem.MolFromSmiles('BrC=CC1OC(C2)(F)C2(Cl)C1')
    >>> isomers = tuple(EnumerateStereoisomers(m))
    >>> len(isomers)
    16
    >>> for smi in sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers):
    ...     print(smi)
    ...
    F[C@@]12C[C@@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@@]12C[C@@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@@]12C[C@@]1(Cl)C[C@H](/C=C/Br)O2
    F[C@@]12C[C@@]1(Cl)C[C@H](/C=C\\Br)O2
    F[C@@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@@]12C[C@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@@]12C[C@]1(Cl)C[C@H](/C=C/Br)O2
    F[C@@]12C[C@]1(Cl)C[C@H](/C=C\\Br)O2
    F[C@]12C[C@@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@]12C[C@@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@]12C[C@@]1(Cl)C[C@H](/C=C/Br)O2
    F[C@]12C[C@@]1(Cl)C[C@H](/C=C\\Br)O2
    F[C@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@]12C[C@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@]12C[C@]1(Cl)C[C@H](/C=C/Br)O2
    F[C@]12C[C@]1(Cl)C[C@H](/C=C\\Br)O2

    Because the molecule is constrained, not all of those isomers can
    actually exist. We can check that:

    >>> opts = StereoEnumerationOptions(tryEmbedding=True)
    >>> isomers = tuple(EnumerateStereoisomers(m, options=opts))
    >>> len(isomers)
    8
    >>> for smi in sorted(Chem.MolToSmiles(x,isomericSmiles=True) for x in isomers):
    ...     print(smi)
    ...
    F[C@@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@@]12C[C@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@@]12C[C@]1(Cl)C[C@H](/C=C/Br)O2
    F[C@@]12C[C@]1(Cl)C[C@H](/C=C\\Br)O2
    F[C@]12C[C@@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@]12C[C@@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@]12C[C@@]1(Cl)C[C@H](/C=C/Br)O2
    F[C@]12C[C@@]1(Cl)C[C@H](/C=C\\Br)O2

    Or we can force the output to only give us unique isomers:

    >>> m = Chem.MolFromSmiles('FC(Cl)C=CC=CC(F)Cl')
    >>> opts = StereoEnumerationOptions(unique=True)
    >>> isomers = tuple(EnumerateStereoisomers(m, options=opts))
    >>> len(isomers)
    10
    >>> for smi in sorted(Chem.MolToSmiles(x,isomericSmiles=True) for x in isomers):
    ...     print(smi)
    ...
    F[C@@H](Cl)/C=C/C=C/[C@@H](F)Cl
    F[C@@H](Cl)/C=C\\C=C/[C@@H](F)Cl
    F[C@@H](Cl)/C=C\\C=C\\[C@@H](F)Cl
    F[C@H](Cl)/C=C/C=C/[C@@H](F)Cl
    F[C@H](Cl)/C=C/C=C/[C@H](F)Cl
    F[C@H](Cl)/C=C/C=C\\[C@@H](F)Cl
    F[C@H](Cl)/C=C\\C=C/[C@@H](F)Cl
    F[C@H](Cl)/C=C\\C=C/[C@H](F)Cl
    F[C@H](Cl)/C=C\\C=C\\[C@@H](F)Cl
    F[C@H](Cl)/C=C\\C=C\\[C@H](F)Cl

    By default the code only expands unspecified stereocenters:

    >>> m = Chem.MolFromSmiles('BrC=C[C@H]1OC(C2)(F)C2(Cl)C1')
    >>> isomers = tuple(EnumerateStereoisomers(m))
    >>> len(isomers)
    8
    >>> for smi in sorted(Chem.MolToSmiles(x,isomericSmiles=True) for x in isomers):
    ...     print(smi)
    ...
    F[C@@]12C[C@@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@@]12C[C@@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@@]12C[C@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@]12C[C@@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@]12C[C@@]1(Cl)C[C@@H](/C=C\\Br)O2
    F[C@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2
    F[C@]12C[C@]1(Cl)C[C@@H](/C=C\\Br)O2

    But we can change that behavior:

    >>> opts = StereoEnumerationOptions(onlyUnassigned=False)
    >>> isomers = tuple(EnumerateStereoisomers(m, options=opts))
    >>> len(isomers)
    16

    Since the result is a generator, we can allow exploring at least parts of very
    large result sets:

    >>> m = Chem.MolFromSmiles('Br' + '[CH](Cl)' * 20 + 'F')
    >>> opts = StereoEnumerationOptions(maxIsomers=0)
    >>> isomers = EnumerateStereoisomers(m, options=opts)
    >>> for x in range(5):
    ...   print(Chem.MolToSmiles(next(isomers),isomericSmiles=True))
    F[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)Br
    F[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)Br
    F[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)Br
    F[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)Br
    F[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)Br

    Or randomly sample a small subset. Note that if we want that sampling to be consistent
    across python versions we need to provide a random number seed:

    >>> m = Chem.MolFromSmiles('Br' + '[CH](Cl)' * 20 + 'F')
    >>> opts = StereoEnumerationOptions(maxIsomers=3,rand=0xf00d)
    >>> isomers = EnumerateStereoisomers(m, options=opts)
    >>> for smi in isomers: #sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers):
    ...     print(Chem.MolToSmiles(smi))
    F[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)Br
    F[C@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)Br
    F[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H](Cl)Br

    """
    tm = Chem.Mol(m)
    for atom in tm.GetAtoms():
        atom.ClearProp('_CIPCode')
    for bond in tm.GetBonds():
        if bond.GetBondDir() == Chem.BondDir.EITHERDOUBLE:
            bond.SetBondDir(Chem.BondDir.NONE)
    flippers = _getFlippers(tm, options)
    nCenters = len(flippers)
    if not nCenters:
        yield tm
        return
    if options.maxIsomers == 0 or 2 ** nCenters <= options.maxIsomers:
        bitsource = _RangeBitsGenerator(nCenters)
    else:
        if options.rand is None:
            seed = hash(tuple(sorted([(a.GetDegree(), a.GetAtomicNum()) for a in tm.GetAtoms()])))
            rand = random.Random(seed)
        elif isinstance(options.rand, random.Random):
            rand = options.rand
        else:
            rand = random.Random(options.rand)
        bitsource = _UniqueRandomBitsGenerator(nCenters, options.maxIsomers, rand)
    isomersSeen = set()
    numIsomers = 0
    for bitflag in bitsource:
        for i in range(nCenters):
            flag = bool(bitflag & 1 << i)
            flippers[i].flip(flag)
        if tm.GetStereoGroups():
            isomer = Chem.RWMol(tm)
            isomer.SetStereoGroups([])
        else:
            isomer = Chem.Mol(tm)
        Chem.SetDoubleBondNeighborDirections(isomer)
        isomer.ClearComputedProps(includeRings=False)
        Chem.AssignStereochemistry(isomer, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        if options.unique:
            cansmi = Chem.MolToSmiles(isomer, isomericSmiles=True)
            if cansmi in isomersSeen:
                continue
            isomersSeen.add(cansmi)
        if options.tryEmbedding:
            ntm = Chem.AddHs(isomer)
            cid = EmbedMolecule(ntm, randomSeed=bitflag & 2147483647)
            if cid >= 0:
                conf = Chem.Conformer(isomer.GetNumAtoms())
                for aid in range(isomer.GetNumAtoms()):
                    conf.SetAtomPosition(aid, ntm.GetConformer().GetAtomPosition(aid))
                isomer.AddConformer(conf)
        else:
            cid = 1
        if cid >= 0:
            yield isomer
            numIsomers += 1
            if options.maxIsomers != 0 and numIsomers >= options.maxIsomers:
                break
        elif verbose:
            print('%s    failed to embed' % Chem.MolToSmiles(isomer, isomericSmiles=True))
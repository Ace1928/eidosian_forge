import os
import sys
from Chem import AllChem as Chem
def GetFrame(mol, mode='Scaff'):
    """return a ganeric molecule defining the reduced scaffold of the input mol.
    mode can be 'Scaff' or 'RedScaff':

    Scaff	->	chop off the side chains and return the scaffold

    RedScaff	->	remove all linking chains and connect the rings
    directly at the atoms where the linker was
    """
    ring = mol.GetRingInfo()
    RingAtoms = flatten(ring.AtomRings())
    NonRingAtoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in RingAtoms]
    RingNeighbors = []
    Paths = []
    for NonRingAtom in NonRingAtoms:
        for neighbor in mol.GetAtomWithIdx(NonRingAtom).GetNeighbors():
            if neighbor.GetIdx() in RingAtoms:
                RingNeighbors.append(NonRingAtom)
                Paths.append([neighbor.GetIdx(), NonRingAtom])
                break
    PosConnectors = [x for x in NonRingAtoms if x not in RingNeighbors]
    Framework = [x for x in RingAtoms]
    Linkers = []
    while len(Paths) > 0:
        NewPaths = []
        for P in Paths:
            if P is None:
                print('ooh')
            else:
                for neighbor in mol.GetAtomWithIdx(P[-1]).GetNeighbors():
                    if neighbor.GetIdx() not in P:
                        if neighbor.GetIdx() in NonRingAtoms:
                            n = P[:]
                            n.append(neighbor.GetIdx())
                            NewPaths.append(n[:])
                        elif neighbor.GetIdx() in RingAtoms:
                            n = P[:]
                            n.append(neighbor.GetIdx())
                            Linkers.append(n)
                            Framework = Framework + P[:]
        Paths = NewPaths[:]
    if mode == 'RedScaff':
        Framework = list(set(Framework))
        todel = []
        NonRingAtoms.sort(reverse=True)
        em = Chem.EditableMol(mol)
        BondsToAdd = [sorted([i[0], i[-1]]) for i in Linkers]
        mem = []
        for i in BondsToAdd:
            if i not in mem:
                em.AddBond(i[0], i[1], Chem.BondType.SINGLE)
                mem.append(i)
        for i in NonRingAtoms:
            todel.append(i)
        for i in todel:
            em.RemoveAtom(i)
        m = em.GetMol()
        return m
    if mode == 'Scaff':
        Framework = list(set(Framework))
        todel = []
        NonRingAtoms.sort(reverse=True)
        for i in NonRingAtoms:
            if i is not None:
                if i not in Framework:
                    todel.append(i)
        em = Chem.EditableMol(mol)
        for i in todel:
            em.RemoveAtom(i)
        m = em.GetMol()
        return m
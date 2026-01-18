import os
import warnings
from collections import namedtuple
from importlib.util import find_spec
from io import BytesIO
import numpy
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw.rdMolDraw2D import *
def _getRDKitEnv(mol, bondPath, baseRad, aromaticColor, extraColor, nonAromaticColor, **kwargs):
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    atomsToUse = set()
    for b in bondPath:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    amap = {}
    submol = Chem.PathToSubmol(mol, bondPath, atomMap=amap)
    Chem.FastFindRings(submol)
    conf = Chem.Conformer(submol.GetNumAtoms())
    confOri = mol.GetConformer(0)
    for i1, i2 in amap.items():
        conf.SetAtomPosition(i2, confOri.GetAtomPosition(i1))
    submol.AddConformer(conf)
    envSubmol = []
    for i1, i2 in amap.items():
        for b in bondPath:
            beginAtom = amap[mol.GetBondWithIdx(b).GetBeginAtomIdx()]
            endAtom = amap[mol.GetBondWithIdx(b).GetEndAtomIdx()]
            envSubmol.append(submol.GetBondBetweenAtoms(beginAtom, endAtom).GetIdx())
    atomcolors, bondcolors = ({}, {})
    highlightAtoms, highlightBonds = ([], [])
    highlightRadii = {}
    for aidx in amap.keys():
        if aidx in atomsToUse:
            color = None
            if aromaticColor and mol.GetAtomWithIdx(aidx).GetIsAromatic():
                color = aromaticColor
            elif nonAromaticColor and (not mol.GetAtomWithIdx(aidx).GetIsAromatic()):
                color = nonAromaticColor
            if color is not None:
                atomcolors[amap[aidx]] = color
                highlightAtoms.append(amap[aidx])
                highlightRadii[amap[aidx]] = baseRad
    color = extraColor
    for bid in submol.GetBonds():
        bidx = bid.GetIdx()
        if bidx not in envSubmol:
            bondcolors[bidx] = color
            highlightBonds.append(bidx)
    return FingerprintEnv(submol, highlightAtoms, atomcolors, highlightBonds, bondcolors, highlightRadii)
import copy
import functools
import math
import numpy
from rdkit import Chem
def AddMol(self, mol, centerIt=True, molTrans=None, drawingTrans=None, highlightAtoms=[], confId=-1, flagCloseContactsDist=2, highlightMap=None, ignoreHs=False, highlightBonds=[], **kwargs):
    """Set the molecule to be drawn.

    Parameters:
      hightlightAtoms -- list of atoms to highlight (default [])
      highlightMap -- dictionary of (atom, color) pairs (default None)

    Notes:
      - specifying centerIt will cause molTrans and drawingTrans to be ignored
    """
    conf = mol.GetConformer(confId)
    if 'coordScale' in kwargs:
        self.drawingOptions.coordScale = kwargs['coordScale']
    self.currDotsPerAngstrom = self.drawingOptions.dotsPerAngstrom
    self.currAtomLabelFontSize = self.drawingOptions.atomLabelFontSize
    if centerIt:
        self.scaleAndCenter(mol, conf, ignoreHs=ignoreHs)
    else:
        self.molTrans = molTrans or (0, 0)
        self.drawingTrans = drawingTrans or (0, 0)
    font = Font(face=self.drawingOptions.atomLabelFontFace, size=self.currAtomLabelFontSize)
    obds = None
    if not mol.HasProp('_drawingBondsWedged'):
        obds = [x.GetBondDir() for x in mol.GetBonds()]
        Chem.WedgeMolBonds(mol, conf)
    includeAtomNumbers = kwargs.get('includeAtomNumbers', self.drawingOptions.includeAtomNumbers)
    self.atomPs[mol] = {}
    self.boundingBoxes[mol] = [0] * 4
    self.activeMol = mol
    self.bondRings = mol.GetRingInfo().BondRings()
    labelSizes = {}
    for atom in mol.GetAtoms():
        labelSizes[atom.GetIdx()] = None
        if ignoreHs and atom.GetAtomicNum() == 1:
            drawAtom = False
        else:
            drawAtom = True
        idx = atom.GetIdx()
        pos = self.atomPs[mol].get(idx, None)
        if pos is None:
            pos = self.transformPoint(conf.GetAtomPosition(idx) * self.drawingOptions.coordScale)
            self.atomPs[mol][idx] = pos
            if drawAtom:
                self.boundingBoxes[mol][0] = min(self.boundingBoxes[mol][0], pos[0])
                self.boundingBoxes[mol][1] = min(self.boundingBoxes[mol][1], pos[1])
                self.boundingBoxes[mol][2] = max(self.boundingBoxes[mol][2], pos[0])
                self.boundingBoxes[mol][3] = max(self.boundingBoxes[mol][3], pos[1])
        if not drawAtom:
            continue
        nbrSum = [0, 0]
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if ignoreHs and nbr.GetAtomicNum() == 1:
                continue
            nbrIdx = nbr.GetIdx()
            if nbrIdx > idx:
                nbrPos = self.atomPs[mol].get(nbrIdx, None)
                if nbrPos is None:
                    nbrPos = self.transformPoint(conf.GetAtomPosition(nbrIdx) * self.drawingOptions.coordScale)
                    self.atomPs[mol][nbrIdx] = nbrPos
                    self.boundingBoxes[mol][0] = min(self.boundingBoxes[mol][0], nbrPos[0])
                    self.boundingBoxes[mol][1] = min(self.boundingBoxes[mol][1], nbrPos[1])
                    self.boundingBoxes[mol][2] = max(self.boundingBoxes[mol][2], nbrPos[0])
                    self.boundingBoxes[mol][3] = max(self.boundingBoxes[mol][3], nbrPos[1])
            else:
                nbrPos = self.atomPs[mol][nbrIdx]
            nbrSum[0] += nbrPos[0] - pos[0]
            nbrSum[1] += nbrPos[1] - pos[1]
        iso = atom.GetIsotope()
        labelIt = not self.drawingOptions.noCarbonSymbols or iso or atom.GetAtomicNum() != 6 or (atom.GetFormalCharge() != 0) or atom.GetNumRadicalElectrons() or includeAtomNumbers or atom.HasProp('molAtomMapNumber') or (atom.GetDegree() == 0)
        orient = ''
        if labelIt:
            baseOffset = 0
            if includeAtomNumbers:
                symbol = str(atom.GetIdx())
                symbolLength = len(symbol)
            else:
                base = atom.GetSymbol()
                if base == 'H' and (iso == 2 or iso == 3) and self.drawingOptions.atomLabelDeuteriumTritium:
                    if iso == 2:
                        base = 'D'
                    else:
                        base = 'T'
                    iso = 0
                symbolLength = len(base)
                nHs = 0
                if not atom.HasQuery():
                    nHs = atom.GetTotalNumHs()
                hs = ''
                if nHs == 1:
                    hs = 'H'
                    symbolLength += 1
                elif nHs > 1:
                    hs = 'H<sub>%d</sub>' % nHs
                    symbolLength += 1 + len(str(nHs))
                chg = atom.GetFormalCharge()
                if chg == 0:
                    chg = ''
                elif chg == 1:
                    chg = '+'
                elif chg == -1:
                    chg = '-'
                else:
                    chg = '%+d' % chg
                symbolLength += len(chg)
                if chg:
                    chg = '<sup>%s</sup>' % chg
                if atom.GetNumRadicalElectrons():
                    rad = self.drawingOptions.radicalSymbol * atom.GetNumRadicalElectrons()
                    rad = '<sup>%s</sup>' % rad
                    symbolLength += atom.GetNumRadicalElectrons()
                else:
                    rad = ''
                isotope = ''
                isotopeLength = 0
                if iso:
                    isotope = '<sup>%d</sup>' % atom.GetIsotope()
                    isotopeLength = len(str(atom.GetIsotope()))
                    symbolLength += isotopeLength
                mapNum = ''
                mapNumLength = 0
                if atom.HasProp('molAtomMapNumber'):
                    mapNum = ':' + atom.GetProp('molAtomMapNumber')
                    mapNumLength = 1 + len(str(atom.GetProp('molAtomMapNumber')))
                    symbolLength += mapNumLength
                deg = atom.GetDegree()
                if deg == 0:
                    tSym = periodicTable.GetElementSymbol(atom.GetAtomicNum())
                    if tSym in ('O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'At'):
                        symbol = '%s%s%s%s%s%s' % (hs, isotope, base, chg, rad, mapNum)
                    else:
                        symbol = '%s%s%s%s%s%s' % (isotope, base, hs, chg, rad, mapNum)
                elif deg > 1 or nbrSum[0] < 1:
                    symbol = '%s%s%s%s%s%s' % (isotope, base, hs, chg, rad, mapNum)
                    baseOffset = 0.5 - (isotopeLength + len(base) / 2.0) / symbolLength
                else:
                    symbol = '%s%s%s%s%s%s' % (rad, chg, hs, isotope, base, mapNum)
                    baseOffset = -0.5 + (mapNumLength + len(base) / 2.0) / symbolLength
                if deg == 1:
                    if abs(nbrSum[1]) > 1:
                        islope = nbrSum[0] / abs(nbrSum[1])
                    else:
                        islope = nbrSum[0]
                    if abs(islope) > 0.3:
                        if islope > 0:
                            orient = 'W'
                        else:
                            orient = 'E'
                    elif abs(nbrSum[1]) > 10:
                        if nbrSum[1] > 0:
                            orient = 'N'
                        else:
                            orient = 'S'
                else:
                    orient = 'C'
            if highlightMap and idx in highlightMap:
                color = highlightMap[idx]
            elif highlightAtoms and idx in highlightAtoms:
                color = self.drawingOptions.selectColor
            else:
                color = self.drawingOptions.elemDict.get(atom.GetAtomicNum(), (0, 0, 0))
            labelSize = self._drawLabel(symbol, pos, baseOffset, font, color=color, orientation=orient)
            labelSizes[atom.GetIdx()] = [labelSize, orient]
    for bond in mol.GetBonds():
        atom, idx = (bond.GetBeginAtom(), bond.GetBeginAtomIdx())
        nbr, nbrIdx = (bond.GetEndAtom(), bond.GetEndAtomIdx())
        pos = self.atomPs[mol].get(idx, None)
        nbrPos = self.atomPs[mol].get(nbrIdx, None)
        if highlightBonds and bond.GetIdx() in highlightBonds:
            width = 2.0 * self.drawingOptions.bondLineWidth
            color = self.drawingOptions.selectColor
            color2 = self.drawingOptions.selectColor
        elif highlightAtoms and idx in highlightAtoms and (nbrIdx in highlightAtoms):
            width = 2.0 * self.drawingOptions.bondLineWidth
            color = self.drawingOptions.selectColor
            color2 = self.drawingOptions.selectColor
        elif highlightMap is not None and idx in highlightMap and (nbrIdx in highlightMap):
            width = 2.0 * self.drawingOptions.bondLineWidth
            color = highlightMap[idx]
            color2 = highlightMap[nbrIdx]
        else:
            width = self.drawingOptions.bondLineWidth
            if self.drawingOptions.colorBonds:
                color = self.drawingOptions.elemDict.get(atom.GetAtomicNum(), (0, 0, 0))
                color2 = self.drawingOptions.elemDict.get(nbr.GetAtomicNum(), (0, 0, 0))
            else:
                color = self.drawingOptions.defaultColor
                color2 = color
        self._drawBond(bond, atom, nbr, pos, nbrPos, conf, color=color, width=width, color2=color2, labelSize1=labelSizes[idx], labelSize2=labelSizes[nbrIdx])
    if obds:
        for i, d in enumerate(obds):
            mol.GetBondWithIdx(i).SetBondDir(d)
    if flagCloseContactsDist > 0:
        tol = flagCloseContactsDist * flagCloseContactsDist
        for i, _ in enumerate(mol.GetAtoms()):
            pi = numpy.array(self.atomPs[mol][i])
            for j in range(i + 1, mol.GetNumAtoms()):
                pj = numpy.array(self.atomPs[mol][j])
                d = pj - pi
                dist2 = d[0] * d[0] + d[1] * d[1]
                if dist2 <= tol:
                    self.canvas.addCanvasPolygon(((pi[0] - 2 * flagCloseContactsDist, pi[1] - 2 * flagCloseContactsDist), (pi[0] + 2 * flagCloseContactsDist, pi[1] - 2 * flagCloseContactsDist), (pi[0] + 2 * flagCloseContactsDist, pi[1] + 2 * flagCloseContactsDist), (pi[0] - 2 * flagCloseContactsDist, pi[1] + 2 * flagCloseContactsDist)), color=(1.0, 0, 0), fill=False, stroke=True)
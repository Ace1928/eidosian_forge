import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def MolsToHTMLTable(mols, molsPerRow=3, subImgSize=(200, 200), legends=None, highlightAtomLists=None, highlightBondLists=None, useSVG=False, drawOptions=None, **kwargs):
    if legends and len(legends) > len(mols):
        legends = legends[:len(mols)]
    if highlightAtomLists and len(highlightAtomLists) > len(mols):
        highlightAtomLists = highlightAtomLists[:len(mols)]
    if highlightBondLists and len(highlightBondLists) > len(mols):
        highlightBondLists = highlightBondLists[:len(mols)]
    nRows = (len(mols) - 1) // molsPerRow + 1 if mols else 0
    if nRows:
        doc = minidom.Document()
        table = doc.createElement('table')
    res = ''
    i = 0
    for _ in range(nRows):
        tr = doc.createElement('tr')
        for _ in range(molsPerRow):
            td = doc.createElement('td')
            td.setAttribute('style', f'padding: 0; background-color: white;')
            highlights = None
            mol = mols[i]
            legend = legends[i] if legends else None
            highlights = highlightAtomLists[i] if highlightAtomLists else None
            kwargs['highlightBonds'] = highlightBondLists[i] if highlightBondLists else None
            content = None
            if isinstance(mol, Chem.Mol):
                if isEnabled(mol):
                    content = generateHTMLBody(mol, subImgSize, drawOptions=drawOptions, legend=legend, useSVG=useSVG, highlightAtoms=highlights, **kwargs)
                else:
                    fn = Draw._moltoSVG if useSVG else Draw._moltoimg
                    content = fn(mol, subImgSize, highlights, legend, **kwargs)
            try:
                content = minidom.parseString(content)
                td.appendChild(content.firstChild)
            except Exception as e:
                log.warning('Failed to parse HTML returned by generateHTMLBody()')
            tr.appendChild(td)
            i += 1
            if i == len(mols):
                break
        table.appendChild(tr)
    if nRows:
        doc.appendChild(table)
        res = doc.toxml()
    return injectHTMLFooterAfterTable(res)
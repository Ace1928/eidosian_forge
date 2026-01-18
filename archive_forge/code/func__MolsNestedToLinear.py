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
def _MolsNestedToLinear(molsMatrix, legendsMatrix=None, highlightAtomListsMatrix=None, highlightBondListsMatrix=None):
    """Converts a nested data structure (where each data substructure represents a row in mol grid image)
  to a linear one, padding rows as needed so all rows are the length of the longest row
  """
    nMolsRows = len(molsMatrix)
    if legendsMatrix is not None:
        nLegendsRows = len(legendsMatrix)
        if nLegendsRows != nMolsRows:
            err = f'If legendsMatrix is provided it must be the same length (have the same number '
            err += f'of sub-iterables) as molsMatrix, {nMolsRows}; its length is {nLegendsRows}.'
            raise ValueError(err)
        for rowIndex, row in enumerate(legendsMatrix):
            if len(row) != len(molsMatrix[rowIndex]):
                err = f'If legendsMatrix is provided each of its sub-iterables must be the same length '
                err += f'as the corresponding sub-iterable of molsMatrix. For sub-iterable of index '
                err += f'{rowIndex}, its length in molsMatrix is {len(molsMatrix[rowIndex])} '
                err += f'while its length in legendsMatrix is {len(row)}.'
                raise ValueError(err)
    if highlightAtomListsMatrix is not None:
        nHighlightAtomListsRows = len(highlightAtomListsMatrix)
        if nHighlightAtomListsRows != nMolsRows:
            err = f'If highlightAtomListsMatrix is provided it must be the same length (have the same number '
            err += f'of sub-iterables) as molsMatrix, {nMolsRows}; its length is {nHighlightAtomListsRows}.'
            raise ValueError(err)
        for rowIndex, row in enumerate(highlightAtomListsMatrix):
            if len(row) != len(molsMatrix[rowIndex]):
                err = f'If highlightAtomListsMatrix is provided each of its sub-iterables must be the same length '
                err += f'as the corresponding sub-iterable of molsMatrix. For sub-iterable of index '
                err += f'{rowIndex}, its length in molsMatrix is {len(molsMatrix[rowIndex])} '
                err += f'while its length in highlightAtomListsMatrix is {len(row)}.'
                raise ValueError(err)
    if highlightBondListsMatrix is not None:
        nHighlightBondListsRows = len(highlightBondListsMatrix)
        if nHighlightBondListsRows != nMolsRows:
            err = f'If highlightBondListsMatrix is provided it must be the same length (have the same number '
            err += f'of sub-iterables) as molsMatrix, {nMolsRows}; its length is {nHighlightBondListsRows}.'
            raise ValueError(err)
        for rowIndex, row in enumerate(highlightBondListsMatrix):
            if len(row) != len(molsMatrix[rowIndex]):
                err = f'If highlightBondListsMatrix is provided each of its sub-iterables must be the same length '
                err += f'as the corresponding sub-iterable of molsMatrix. For sub-iterable of index '
                err += f'{rowIndex}, its length in molsMatrix is {len(molsMatrix[rowIndex])} '
                err += f'while its length in highlightBondListsMatrix is {len(row)}.'
                raise ValueError(err)
    molsPerRow = max((len(row) for row in molsMatrix))
    molsMatrixPadded = _padMatrix(molsMatrix, molsPerRow, None)
    mols = _flattenTwoDList(molsMatrixPadded)
    if legendsMatrix is not None:
        legendsMatrixPadded = _padMatrix(legendsMatrix, molsPerRow, '')
        legends = _flattenTwoDList(legendsMatrixPadded)
    else:
        legends = None
    if highlightAtomListsMatrix is not None:
        highlightAtomListsPadded = _padMatrix(highlightAtomListsMatrix, molsPerRow, [])
        highlightAtomLists = _flattenTwoDList(highlightAtomListsPadded)
    else:
        highlightAtomLists = None
    if highlightBondListsMatrix is not None:
        highlightBondListsPadded = _padMatrix(highlightBondListsMatrix, molsPerRow, [])
        highlightBondLists = _flattenTwoDList(highlightBondListsPadded)
    else:
        highlightBondLists = None
    return (mols, molsPerRow, legends, highlightAtomLists, highlightBondLists)
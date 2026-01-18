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
def ShowMol(mol, size=(300, 300), kekulize=True, wedgeBonds=True, title='RDKit Molecule', stayInFront=True, **kwargs):
    """ Generates a picture of a molecule and displays it in a Tkinter window
  """
    import tkinter
    from PIL import ImageTk
    img = MolToImage(mol, size, kekulize, wedgeBonds, **kwargs)
    tkRoot = tkinter.Tk()
    tkRoot.title(title)
    tkPI = ImageTk.PhotoImage(img)
    tkLabel = tkinter.Label(tkRoot, image=tkPI)
    tkLabel.place(x=0, y=0, width=img.size[0], height=img.size[1])
    tkRoot.geometry('%dx%d' % img.size)
    tkRoot.lift()
    if stayInFront:
        tkRoot.attributes('-topmost', True)
    tkRoot.mainloop()
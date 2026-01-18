import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetSimilarityMapFromWeights(mol, weights, colorMap=None, scale=-1, size=(250, 250), sigma=None, coordScale=1.5, step=0.01, colors='k', contourLines=10, alpha=0.5, draw2d=None, **kwargs):
    """
    Generates the similarity map for a molecule given the atomic weights.

    Parameters:
      mol -- the molecule of interest
      colorMap -- the matplotlib color map scheme, default is custom PiWG color map
      scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                            scale = double -> this is the maximum scale
      size -- the size of the figure
      sigma -- the sigma for the Gaussians
      coordScale -- scaling factor for the coordinates
      step -- the step for calcAtomGaussian
      colors -- color of the contour lines
      contourLines -- if integer number N: N contour lines are drawn
                      if list(numbers): contour lines at these numbers are drawn
      alpha -- the alpha blending value for the contour lines
      kwargs -- additional arguments for drawing
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError('too few atoms')
    if draw2d is not None:
        mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)
        if sigma is None:
            if mol.GetNumBonds() > 0:
                bond = mol.GetBondWithIdx(0)
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1) - mol.GetConformer().GetAtomPosition(idx2)).Length()
            else:
                sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0) - mol.GetConformer().GetAtomPosition(1)).Length()
            sigma = round(sigma, 2)
        sigmas = [sigma] * mol.GetNumAtoms()
        locs = []
        for i in range(mol.GetNumAtoms()):
            p = mol.GetConformer().GetAtomPosition(i)
            locs.append(Geometry.Point2D(p.x, p.y))
        draw2d.ClearDrawing()
        ps = Draw.ContourParams()
        ps.fillGrid = True
        ps.gridResolution = 0.1
        ps.extraGridPadding = 0.5
        if colorMap is not None:
            if cm is not None and isinstance(colorMap, type(cm.Blues)):
                clrs = [tuple(x) for x in colorMap([0, 0.5, 1])]
            elif type(colorMap) == str:
                if cm is None:
                    raise ValueError('cannot provide named colormaps unless matplotlib is installed')
                clrs = [tuple(x) for x in cm.get_cmap(colorMap)([0, 0.5, 1])]
            else:
                clrs = [colorMap[0], colorMap[1], colorMap[2]]
            ps.setColourMap(clrs)
        Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)
        draw2d.drawOptions().clearBackground = False
        draw2d.DrawMolecule(mol)
        return draw2d
    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i]) ** 2 for i in range(2)]))
        else:
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i]) ** 2 for i in range(2)]))
        sigma = round(sigma, 2)
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    if scale <= 0.0:
        maxScale = max(math.fabs(numpy.min(z)), math.fabs(numpy.max(z)))
    else:
        maxScale = scale
    if colorMap is None:
        if cm is None:
            raise RuntimeError('matplotlib failed to import')
        PiYG_cmap = cm.get_cmap('PiYG', 2)
        colorMap = LinearSegmentedColormap.from_list('PiWG', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=255)
    fig.axes[0].imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1), vmin=-maxScale, vmax=maxScale)
    if len([w for w in weights if w != 0.0]):
        contourset = fig.axes[0].contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
        for j, c in enumerate(contourset.collections):
            if contourset.levels[j] == 0.0:
                c.set_linewidth(0.0)
            elif contourset.levels[j] < 0:
                c.set_dashes([(0, (3.0, 3.0))])
    fig.axes[0].set_axis_off()
    return fig
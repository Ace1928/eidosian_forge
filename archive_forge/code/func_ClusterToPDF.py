import numpy
from . import ClusterUtils
def ClusterToPDF(cluster, fileName, size=(300, 300), ptColors=[], lineWidth=None, showIndices=0, stopAtCentroids=0, logScale=0):
    """ handles the work of drawing a cluster tree to an PDF file

    **Arguments**

      - cluster: the cluster tree to be drawn

      - fileName: the name of the file to be created

      - size: the size of output canvas

      - ptColors: if this is specified, the _colors_ will be used to color
        the terminal nodes of the cluster tree.  (color == _pid.Color_)

      - lineWidth: if specified, it will be used for the widths of the lines
        used to draw the tree

   **Notes**

     - if _ptColors_ is the wrong length for the number of possible terminal
       node types, this will throw an IndexError

     - terminal node types are determined using their _GetData()_ methods

  """
    try:
        from rdkit.sping.PDF import pidPDF
    except ImportError:
        from rdkit.piddle import piddlePDF
        pidPDF = piddlePDF
    canvas = pidPDF.PDFCanvas(size, fileName)
    if lineWidth is None:
        lineWidth = VisOpts.lineWidth
    DrawClusterTree(cluster, canvas, size, ptColors=ptColors, lineWidth=lineWidth, showIndices=showIndices, stopAtCentroids=stopAtCentroids, logScale=logScale)
    if fileName:
        canvas.save()
    return canvas
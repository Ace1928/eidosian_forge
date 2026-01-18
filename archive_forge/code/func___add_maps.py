import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_maps(self):
    """Add maps to the drawing of the map (PRIVATE).

        We do this first, as they're regional labels to be overlaid by
        information.  Also, we want to set the color to something subtle.

        We're using Hex colors because that's what KGML uses, and
        Reportlab doesn't mind.
        """
    for m in self.pathway.maps:
        for g in m.graphics:
            self.drawing.setStrokeColor('#888888')
            self.drawing.setFillColor('#DDDDDD')
            self.__add_graphics(g)
            if self.label_maps:
                self.drawing.setFillColor('#888888')
                self.__add_labels(g)
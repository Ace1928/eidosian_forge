import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_genes(self):
    """Add gene elements to the drawing of the map (PRIVATE)."""
    for gene in self.pathway.genes:
        for g in gene.graphics:
            self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
            self.drawing.setFillColor(color_to_reportlab(g.bgcolor))
            self.__add_graphics(g)
            if self.label_compounds:
                self.drawing.setFillColor(darken(g.fgcolor))
                self.__add_labels(g)
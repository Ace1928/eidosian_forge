import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def __add_relations(self):
    """Add relations to the map (PRIVATE).

        This is tricky. There is no defined graphic in KGML for a
        relation, and the corresponding entries are typically defined
        as objects 'to be connected somehow'.  KEGG uses KegSketch, which
        is not public, and most third-party software draws straight line
        arrows, with heads to indicate the appropriate direction
        (at both ends for reversible reactions), using solid lines for
        ECrel relation types, and dashed lines for maplink relation types.

        The relation has:
        - entry1: 'from' node
        - entry2: 'to' node
        - subtype: what the relation refers to

        Typically we have entry1 = map/ortholog; entry2 = map/ortholog,
        subtype = compound.
        """
    for relation in list(self.pathway.relations):
        if relation.type == 'maplink':
            self.drawing.setDash(6, 3)
        else:
            self.drawing.setDash()
        for s in relation.subtypes:
            subtype = self.pathway.entries[s[1]]
            self.__draw_arrow(relation.entry1, subtype)
            self.__draw_arrow(subtype, relation.entry2)
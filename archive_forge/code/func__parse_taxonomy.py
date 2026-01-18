from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _parse_taxonomy(self, parent):
    """Parse taxonomic information for a clade (PRIVATE)."""
    taxonomy = PX.Taxonomy(**parent.attrib)
    for event, elem in self.context:
        namespace, tag = _split_namespace(elem.tag)
        if event == 'end':
            if tag == 'taxonomy':
                parent.clear()
                break
            if tag in ('id', 'uri'):
                setattr(taxonomy, tag, getattr(self, tag)(elem))
            elif tag == 'common_name':
                taxonomy.common_names.append(_collapse_wspace(elem.text))
            elif tag == 'synonym':
                taxonomy.synonyms.append(elem.text)
            elif tag in ('code', 'scientific_name', 'authority', 'rank'):
                setattr(taxonomy, tag, elem.text)
            elif namespace != NAMESPACES['phy']:
                taxonomy.other.append(self.other(elem, namespace, tag))
                parent.clear()
    return taxonomy
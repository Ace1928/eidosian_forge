from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _parse_clade(self, parent):
    """Parse a Clade node and its children, recursively (PRIVATE)."""
    clade = PX.Clade(**parent.attrib)
    if clade.branch_length is not None:
        clade.branch_length = float(clade.branch_length)
    tag_stack = []
    for event, elem in self.context:
        namespace, tag = _split_namespace(elem.tag)
        if event == 'start':
            if tag == 'clade':
                clade.clades.append(self._parse_clade(elem))
                continue
            if tag == 'taxonomy':
                clade.taxonomies.append(self._parse_taxonomy(elem))
                continue
            if tag == 'sequence':
                clade.sequences.append(self._parse_sequence(elem))
                continue
            if tag in self._clade_tracked_tags:
                tag_stack.append(tag)
        if event == 'end':
            if tag == 'clade':
                elem.clear()
                break
            if tag != tag_stack[-1]:
                continue
            tag_stack.pop()
            if tag in self._clade_list_types:
                getattr(clade, self._clade_list_types[tag]).append(getattr(self, tag)(elem))
            elif tag in self._clade_complex_types:
                setattr(clade, tag, getattr(self, tag)(elem))
            elif tag == 'branch_length':
                if clade.branch_length is not None:
                    raise PhyloXMLError('Attribute branch_length was already set for this Clade.')
                clade.branch_length = _float(elem.text)
            elif tag == 'width':
                clade.width = _float(elem.text)
            elif tag == 'name':
                clade.name = _collapse_wspace(elem.text)
            elif tag == 'node_id':
                clade.node_id = PX.Id(elem.text.strip(), elem.attrib.get('provider'))
            elif namespace != NAMESPACES['phy']:
                clade.other.append(self.other(elem, namespace, tag))
                elem.clear()
            else:
                raise PhyloXMLError('Misidentified tag: ' + tag)
    return clade
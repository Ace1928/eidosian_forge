import xml.dom.minidom
def _Replacement_writexml(self, writer, indent='', addindent='', newl=''):
    writer.write(indent + '<' + self.tagName)
    attrs = self._get_attributes()
    a_names = sorted(attrs.keys())
    for a_name in a_names:
        writer.write(' %s="' % a_name)
        _Replacement_write_data(writer, attrs[a_name].value, is_attrib=True)
        writer.write('"')
    if self.childNodes:
        writer.write('>%s' % newl)
        for node in self.childNodes:
            node.writexml(writer, indent + addindent, addindent, newl)
        writer.write(f'{indent}</{self.tagName}>{newl}')
    else:
        writer.write('/>%s' % newl)
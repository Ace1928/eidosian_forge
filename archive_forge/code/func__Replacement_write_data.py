import xml.dom.minidom
def _Replacement_write_data(writer, data, is_attrib=False):
    """Writes datachars to writer."""
    data = data.replace('&', '&amp;').replace('<', '&lt;')
    data = data.replace('"', '&quot;').replace('>', '&gt;')
    if is_attrib:
        data = data.replace('\r', '&#xD;').replace('\n', '&#xA;').replace('\t', '&#x9;')
    writer.write(data)
from .atom  import atom_add_entry_type
from .html5 import html5_extra_attributes, remove_rel
class MediaTypes:
    """An enumeration style class: some common media types (better have them at one place to avoid misstyping...)"""
    rdfxml = 'application/rdf+xml'
    turtle = 'text/turtle'
    html = 'text/html'
    xhtml = 'application/xhtml+xml'
    svg = 'application/svg+xml'
    svgi = 'image/svg+xml'
    smil = 'application/smil+xml'
    atom = 'application/atom+xml'
    xml = 'application/xml'
    xmlt = 'text/xml'
    nt = 'text/plain'
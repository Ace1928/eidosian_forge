from docutils import nodes
class math_reference(nodes.Inline, nodes.Referential, nodes.TextElement):
    """A node for a reference for equation."""
    pass
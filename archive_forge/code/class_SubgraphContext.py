from . import lang, files
class SubgraphContext(object):
    """Return a blank instance of the parent and add as subgraph on exit."""

    def __init__(self, parent, kwargs):
        self.parent = parent
        self.graph = parent.__class__(**kwargs)

    def __enter__(self):
        return self.graph

    def __exit__(self, type_, value, traceback):
        if type_ is None:
            self.parent.subgraph(self.graph)
from troveclient import base
class Diagnostics(base.Resource):
    """Account is an opaque instance used to hold account information."""

    def __repr__(self):
        return '<Diagnostics: %s>' % self.version
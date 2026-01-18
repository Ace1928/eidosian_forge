import re
def _normalise(self):
    """Perform normalisation of URI components."""
    self.scheme = self.scheme.lower()
    if self.userinfo is not None:
        self.userinfo = normalise_unreserved(self.userinfo)
    if self.host is not None:
        self.host = normalise_unreserved(self.host.lower())
    if self.port == '':
        self.port = None
    elif self.port is not None:
        if self.port == _default_port.get(self.scheme):
            self.port = None
    if self.host is not None and self.path == '':
        self.path = '/'
    self.path = normalise_unreserved(remove_dot_segments(self.path))
    if self.query is not None:
        self.query = normalise_unreserved(self.query)
    if self.fragment is not None:
        self.fragment = normalise_unreserved(self.fragment)
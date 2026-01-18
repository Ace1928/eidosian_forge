class HttpBoundaryMissing(InvalidHttpResponse):
    """A multipart response ends with no boundary marker.

    This is a special case caused by buggy proxies, described in
    <https://bugs.launchpad.net/bzr/+bug/198646>.
    """
    _fmt = 'HTTP MIME Boundary missing for %(path)s: %(msg)s'

    def __init__(self, path, msg):
        InvalidHttpResponse.__init__(self, path, msg)
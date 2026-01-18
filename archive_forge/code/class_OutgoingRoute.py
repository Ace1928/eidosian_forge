import logging
from time import gmtime
class OutgoingRoute(object):
    """Holds state about a route that is queued for being sent to a given sink.
    """
    __slots__ = ('_path', '_for_route_refresh', 'sink', 'next_outgoing_route', 'prev_outgoing_route', 'next_sink_out_route', 'prev_sink_out_route')

    def __init__(self, path, for_route_refresh=False):
        assert path
        self.sink = None
        self._path = path
        self._for_route_refresh = for_route_refresh

    @property
    def path(self):
        return self._path

    @property
    def for_route_refresh(self):
        return self._for_route_refresh

    def __str__(self):
        return 'OutgoingRoute(path: %s, for_route_refresh: %s)' % (self.path, self.for_route_refresh)
from . import compat
from . import normalizers
from . import uri
from . import uri_reference
def add_fragment(self, fragment):
    """Add a fragment to the URI.

        .. code-block:: python

            >>> URIBuilder().add_fragment('section-2.6.1')
            URIBuilder(scheme=None, userinfo=None, host=None, port=None,
                    path=None, query=None, fragment='section-2.6.1')

        """
    return URIBuilder(scheme=self.scheme, userinfo=self.userinfo, host=self.host, port=self.port, path=self.path, query=self.query, fragment=normalizers.normalize_fragment(fragment))
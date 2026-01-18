import logging
from google.appengine.ext import ndb
from oauth2client import client
class SiteXsrfSecretKeyNDB(ndb.Model):
    """NDB Model for storage for the sites XSRF secret key.

    Since this model uses the same kind as SiteXsrfSecretKey, it can be
    used interchangeably. This simply provides an NDB model for interacting
    with the same data the DB model interacts with.

    There should only be one instance stored of this model, the one used
    for the site.
    """
    secret = ndb.StringProperty()

    @classmethod
    def _get_kind(cls):
        """Return the kind name for this class."""
        return 'SiteXsrfSecretKey'
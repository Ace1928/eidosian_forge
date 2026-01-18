import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen
from breezy.errors import BzrError
from breezy.trace import note
from breezy.urlutils import InvalidURL
class PypiProjectWithoutRepositoryURL(InvalidURL):
    _fmt = 'No repository URL set for pypi project %(name)s'

    def __init__(self, name, url=None):
        BzrError.__init__(self, name=name, url=url)
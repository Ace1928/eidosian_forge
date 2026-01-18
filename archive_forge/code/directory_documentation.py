import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen
from breezy.errors import BzrError
from breezy.trace import note
from breezy.urlutils import InvalidURL
See DirectoryService.look_up
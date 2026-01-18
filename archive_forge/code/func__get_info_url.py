import re
import urllib
from openstack import exceptions
from openstack import resource
from openstack import utils
def _get_info_url(self, url):
    URI_PATTERN_VERSION = re.compile('\\/v\\d+\\.?\\d*(\\/.*)?')
    scheme, netloc, path, params, query, fragment = urllib.parse.urlparse(url)
    if URI_PATTERN_VERSION.search(path):
        path = URI_PATTERN_VERSION.sub('/info', path)
    else:
        path = utils.urljoin(path, 'info')
    return urllib.parse.urlunparse((scheme, netloc, path, params, query, fragment))
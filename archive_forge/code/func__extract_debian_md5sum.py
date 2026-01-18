from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def _extract_debian_md5sum(rev):
    if 'deb-md5' in rev.properties:
        yield ('debian-md5sum', rev.properties['deb-md5'])
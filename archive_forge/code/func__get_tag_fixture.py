import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def _get_tag_fixture(tag_name, **kwargs):
    tag = {'name': tag_name}
    tag.update(kwargs)
    return tag
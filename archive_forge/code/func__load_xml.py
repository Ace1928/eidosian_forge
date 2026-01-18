import os
from lxml import etree
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def _load_xml(filename):
    with open(XMLDIR + filename, 'rb') as f:
        return f.read()
import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
def _str_from_xml(xml, path):
    li = xml.xpath(path, namespaces=_XML_NAMESPACES)
    if len(li) != 1:
        raise IndexError('%s should provide a single element list' % path)
    return li[0]
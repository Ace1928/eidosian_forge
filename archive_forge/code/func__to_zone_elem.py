import copy
import base64
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib
from libcloud.utils.xml import findall, findtext
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
def _to_zone_elem(self, domain=None, type=None, ttl=None, extra=None):
    zone_elem = ET.Element('zone', {})
    if domain:
        domain_elem = ET.SubElement(zone_elem, 'domain')
        domain_elem.text = domain
    if type:
        ns_type_elem = ET.SubElement(zone_elem, 'ns-type')
        if type == 'master':
            ns_type_elem.text = 'pri_sec'
        elif type == 'slave':
            if not extra or 'ns1' not in extra:
                raise LibcloudError('ns1 extra attribute is required ' + 'when zone type is slave', driver=self)
            ns_type_elem.text = 'sec'
            ns1_elem = ET.SubElement(zone_elem, 'ns1')
            ns1_elem.text = extra['ns1']
        elif type == 'std_master':
            if not extra or 'slave-nameservers' not in extra:
                raise LibcloudError('slave-nameservers extra ' + 'attribute is required whenzone ' + 'type is std_master', driver=self)
            ns_type_elem.text = 'pri'
            slave_nameservers_elem = ET.SubElement(zone_elem, 'slave-nameservers')
            slave_nameservers_elem.text = extra['slave-nameservers']
    if ttl:
        default_ttl_elem = ET.SubElement(zone_elem, 'default-ttl')
        default_ttl_elem.text = str(ttl)
    if extra and 'tag-list' in extra:
        tags = extra['tag-list']
        tags_elem = ET.SubElement(zone_elem, 'tag-list')
        tags_elem.text = ' '.join(tags)
    return zone_elem
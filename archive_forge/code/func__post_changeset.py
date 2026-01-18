import copy
import hmac
import uuid
import base64
import datetime
from hashlib import sha1
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib, urlencode
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.aws import AWSGenericResponse, AWSTokenConnection
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError
def _post_changeset(self, zone, changes_list):
    attrs = {'xmlns': NAMESPACE}
    changeset = ET.Element('ChangeResourceRecordSetsRequest', attrs)
    batch = ET.SubElement(changeset, 'ChangeBatch')
    changes = ET.SubElement(batch, 'Changes')
    for action, name, type_, data, extra in changes_list:
        change = ET.SubElement(changes, 'Change')
        ET.SubElement(change, 'Action').text = action
        rrs = ET.SubElement(change, 'ResourceRecordSet')
        if name:
            record_name = name + '.' + zone.domain
        else:
            record_name = zone.domain
        ET.SubElement(rrs, 'Name').text = record_name
        ET.SubElement(rrs, 'Type').text = self.RECORD_TYPE_MAP[type_]
        ET.SubElement(rrs, 'TTL').text = str(extra.get('ttl', '0'))
        rrecs = ET.SubElement(rrs, 'ResourceRecords')
        rrec = ET.SubElement(rrecs, 'ResourceRecord')
        if 'priority' in extra:
            data = '{} {}'.format(extra['priority'], data)
        ET.SubElement(rrec, 'Value').text = data
    uri = API_ROOT + 'hostedzone/' + zone.id + '/rrset'
    data = ET.tostring(changeset)
    self.connection.set_context({'zone_id': zone.id})
    response = self.connection.request(uri, method='POST', data=data)
    return response.status == httplib.OK
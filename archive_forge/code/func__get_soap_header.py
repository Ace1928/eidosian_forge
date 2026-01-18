from __future__ import unicode_literals
import base64
import uuid
import xml.etree.ElementTree as ET
import xmltodict
from six import text_type
from winrm.transport import Transport
from winrm.exceptions import WinRMError, WinRMTransportError, WinRMOperationTimeoutError
def _get_soap_header(self, action=None, resource_uri=None, shell_id=None, message_id=None):
    if not message_id:
        message_id = uuid.uuid4()
    header = {'@xmlns:xsd': 'http://www.w3.org/2001/XMLSchema', '@xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance', '@xmlns:env': xmlns['soapenv'], '@xmlns:a': xmlns['soapaddr'], '@xmlns:b': 'http://schemas.dmtf.org/wbem/wsman/1/cimbinding.xsd', '@xmlns:n': 'http://schemas.xmlsoap.org/ws/2004/09/enumeration', '@xmlns:x': 'http://schemas.xmlsoap.org/ws/2004/09/transfer', '@xmlns:w': 'http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd', '@xmlns:p': 'http://schemas.microsoft.com/wbem/wsman/1/wsman.xsd', '@xmlns:rsp': 'http://schemas.microsoft.com/wbem/wsman/1/windows/shell', '@xmlns:cfg': 'http://schemas.microsoft.com/wbem/wsman/1/config', 'env:Header': {'a:To': 'http://windows-host:5985/wsman', 'a:ReplyTo': {'a:Address': {'@mustUnderstand': 'true', '#text': 'http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous'}}, 'w:MaxEnvelopeSize': {'@mustUnderstand': 'true', '#text': '153600'}, 'a:MessageID': 'uuid:{0}'.format(message_id), 'w:Locale': {'@mustUnderstand': 'false', '@xml:lang': 'en-US'}, 'p:DataLocale': {'@mustUnderstand': 'false', '@xml:lang': 'en-US'}, 'w:OperationTimeout': 'PT{0}S'.format(int(self.operation_timeout_sec)), 'w:ResourceURI': {'@mustUnderstand': 'true', '#text': resource_uri}, 'a:Action': {'@mustUnderstand': 'true', '#text': action}}}
    if shell_id:
        header['env:Header']['w:SelectorSet'] = {'w:Selector': {'@Name': 'ShellId', '#text': shell_id}}
    return header
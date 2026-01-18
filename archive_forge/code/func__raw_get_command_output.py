from __future__ import unicode_literals
import base64
import uuid
import xml.etree.ElementTree as ET
import xmltodict
from six import text_type
from winrm.transport import Transport
from winrm.exceptions import WinRMError, WinRMTransportError, WinRMOperationTimeoutError
def _raw_get_command_output(self, shell_id, command_id):
    req = {'env:Envelope': self._get_soap_header(resource_uri='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd', action='http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Receive', shell_id=shell_id)}
    stream = req['env:Envelope'].setdefault('env:Body', {}).setdefault('rsp:Receive', {}).setdefault('rsp:DesiredStream', {})
    stream['@CommandId'] = command_id
    stream['#text'] = 'stdout stderr'
    res = self.send_message(xmltodict.unparse(req))
    root = ET.fromstring(res)
    stream_nodes = [node for node in root.findall('.//*') if node.tag.endswith('Stream')]
    stdout = stderr = b''
    return_code = -1
    for stream_node in stream_nodes:
        if not stream_node.text:
            continue
        if stream_node.attrib['Name'] == 'stdout':
            stdout += base64.b64decode(stream_node.text.encode('ascii'))
        elif stream_node.attrib['Name'] == 'stderr':
            stderr += base64.b64decode(stream_node.text.encode('ascii'))
    command_done = len([node for node in root.findall('.//*') if node.get('State', '').endswith('CommandState/Done')]) == 1
    if command_done:
        return_code = int(next((node for node in root.findall('.//*') if node.tag.endswith('ExitCode'))).text)
    return (stdout, stderr, return_code, command_done)
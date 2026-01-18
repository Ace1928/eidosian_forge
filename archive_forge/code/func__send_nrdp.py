from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import open_url
from ansible.plugins.callback import CallbackBase
def _send_nrdp(self, state, msg):
    """
        nrpd service check send XMLDATA like this:
        <?xml version='1.0'?>
            <checkresults>
                <checkresult type='service'>
                    <hostname>somehost</hostname>
                    <servicename>someservice</servicename>
                    <state>1</state>
                    <output>WARNING: Danger Will Robinson!|perfdata</output>
                </checkresult>
            </checkresults>
        """
    xmldata = "<?xml version='1.0'?>\n"
    xmldata += '<checkresults>\n'
    xmldata += "<checkresult type='service'>\n"
    xmldata += '<hostname>%s</hostname>\n' % self.hostname
    xmldata += '<servicename>%s</servicename>\n' % self.servicename
    xmldata += '<state>%d</state>\n' % state
    xmldata += '<output>%s</output>\n' % msg
    xmldata += '</checkresult>\n'
    xmldata += '</checkresults>\n'
    body = {'cmd': 'submitcheck', 'token': self.token, 'XMLDATA': to_bytes(xmldata)}
    try:
        response = open_url(self.url, data=urlencode(body), method='POST', validate_certs=self.validate_nrdp_certs)
        return response.read()
    except Exception as ex:
        self._display.warning('NRDP callback cannot send result {0}'.format(ex))
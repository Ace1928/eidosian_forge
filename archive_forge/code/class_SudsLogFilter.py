import http.client as httplib
import io
import logging
import netaddr
from oslo_utils import timeutils
from oslo_utils import uuidutils
import requests
import suds
from suds import cache
from suds import client
from suds import plugin
import suds.sax.element as element
from suds import transport
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class SudsLogFilter(logging.Filter):
    """Filter to mask/truncate vCenter credentials in suds logs."""

    def filter(self, record):
        if not hasattr(record.msg, 'childAtPath'):
            return True
        login = record.msg.childAtPath('/Envelope/Body/Login') or record.msg.childAtPath('/Envelope/Body/SessionIsActive')
        if login is None:
            return True
        if login.childAtPath('userName') is not None:
            login.childAtPath('userName').setText('***')
        if login.childAtPath('password') is not None:
            login.childAtPath('password').setText('***')
        session_id = login.childAtPath('sessionID')
        if session_id is not None:
            session_id.setText(session_id.getText()[-5:])
        return True
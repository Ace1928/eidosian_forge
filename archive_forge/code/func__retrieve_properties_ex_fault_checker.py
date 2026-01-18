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
@staticmethod
def _retrieve_properties_ex_fault_checker(response):
    """Checks the RetrievePropertiesEx API response for errors.

        Certain faults are sent in the SOAP body as a property of missingSet.
        This method raises VimFaultException when a fault is found in the
        response.

        :param response: response from RetrievePropertiesEx API call
        :raises: VimFaultException
        """
    fault_list = []
    details = {}
    if not response:
        LOG.debug('RetrievePropertiesEx API response is empty; setting fault to %s.', exceptions.NOT_AUTHENTICATED)
        fault_list = [exceptions.NOT_AUTHENTICATED]
    else:
        for obj_cont in response.objects:
            if hasattr(obj_cont, 'missingSet'):
                for missing_elem in obj_cont.missingSet:
                    f_type = missing_elem.fault.fault
                    f_name = f_type.__class__.__name__
                    fault_list.append(f_name)
                    if f_name == exceptions.NO_PERMISSION:
                        details['object'] = vim_util.get_moref_value(f_type.object)
                        details['privilegeId'] = f_type.privilegeId
    if fault_list:
        fault_string = _('Error occurred while calling RetrievePropertiesEx.')
        raise exceptions.VimFaultException(fault_list, fault_string, details=details)
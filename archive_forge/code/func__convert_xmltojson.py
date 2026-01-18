from __future__ import (absolute_import, division, print_function)
import os
import json
import time
from ssl import SSLError
from xml.etree import ElementTree as ET
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def _convert_xmltojson(module, job_details, idrac):
    """get all the xml data from PackageList and returns as valid json."""
    data, repo_status, failed_status = ([], False, False)
    try:
        xmldata = ET.fromstring(job_details['PackageList'])
        for iname in xmldata.iter('INSTANCENAME'):
            comp_data = dict([(attr.attrib['NAME'], txt.text) for attr in iname.iter('PROPERTY') for txt in attr])
            component, failed = get_job_status(module, comp_data, idrac)
            if not failed_status and failed:
                failed_status = True
            data.append(component)
        repo_status = True
    except ET.ParseError:
        data = job_details['PackageList']
    return (data, repo_status, failed_status)
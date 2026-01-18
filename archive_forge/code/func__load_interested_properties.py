import os
import re
import shutil
import tarfile
import urllib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _, _LW
def _load_interested_properties(self):
    """Find the OVF properties config file and load it.

        OVF properties config file specifies which metadata of interest to
        extract. Reads in a JSON file named 'ovf-metadata.json' if available.
        See example file at etc/ovf-metadata.json.sample.
        """
    filename = 'ovf-metadata.json'
    match = CONF.find_file(filename)
    if match:
        with open(match, 'r') as properties_file:
            properties = json.loads(properties_file.read())
            self.interested_properties = properties.get('cim_pasd', [])
            if not self.interested_properties:
                msg = _LW('OVF metadata of interest was not specified in ovf-metadata.json config file. Please set "cim_pasd" to a list of interested CIM_ProcessorAllocationSettingData properties.')
                LOG.warning(msg)
    else:
        LOG.warning(_LW('OVF properties config file "ovf-metadata.json" was not found.'))
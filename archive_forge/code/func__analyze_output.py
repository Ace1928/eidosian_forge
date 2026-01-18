import os
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick import utils
def _analyze_output(self, out):
    LOG.debug('Enter into _analyze_output.')
    if out:
        analyse_result = {}
        out_temp = out.split('\n')
        for line in out_temp:
            LOG.debug('Line is %s.', line)
            if line.find('=') != -1:
                key, val = line.split('=', 1)
                LOG.debug('%(key)s = %(val)s', {'key': key, 'val': val})
                if key in ['ret_code', 'ret_desc', 'dev_addr']:
                    analyse_result[key] = val
        return analyse_result
    else:
        return None
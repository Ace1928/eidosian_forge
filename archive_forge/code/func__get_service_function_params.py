import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
def _get_service_function_params(sf_params):
    attrs = {}
    for sf_param in sf_params:
        if 'correlation' in sf_param:
            if sf_param['correlation'] == 'None':
                attrs['correlation'] = None
            else:
                attrs['correlation'] = sf_param['correlation']
        if 'weight' in sf_param:
            attrs['weight'] = sf_param['weight']
    return attrs
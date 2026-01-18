import base64
import binascii
import os
import re
import shlex
from oslo_serialization import jsonutils
from oslo_utils import netutils
from urllib import parse
from urllib import request
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import cliutils as utils
from zunclient import exceptions as exc
from zunclient.i18n import _
def _convert_healthcheck_para(time, err_msg):
    int_pattern = '^\\d+$'
    time_pattern = '^\\d+(s|m|h)$'
    ret = 0
    if re.match(int_pattern, time):
        ret = int(time)
    elif re.match(time_pattern, time):
        if time.endswith('s'):
            ret = int(time.split('s')[0])
        elif time.endswith('m'):
            ret = int(time.split('m')[0]) * 60
        elif time.endswith('h'):
            ret = int(time.split('h')[0]) * 3600
    else:
        raise apiexec.CommandError(err_msg)
    return ret
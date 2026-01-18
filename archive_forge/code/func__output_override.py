import uuid
import base64
from openstackclient.identity import common as identity_common
import os
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
import simplejson as json
import sys
from troveclient.apiclient import exceptions
def _output_override(objs, print_as):
    """Output override flag checking.

    If an output override global flag is set, print with override
    raise BaseException if no printing was overridden.
    """
    if globals().get('json_output', False):
        if print_as == 'list':
            new_objs = []
            for o in objs:
                new_objs.append(o._info)
        elif print_as == 'dict':
            new_objs = objs
        print(json.dumps(new_objs, indent='  '))
    else:
        raise BaseException('No valid output override')
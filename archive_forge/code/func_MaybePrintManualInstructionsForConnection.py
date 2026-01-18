from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from absl import flags
import googleapiclient
import httplib2
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def MaybePrintManualInstructionsForConnection(connection, flag_format=None):
    """Prints follow-up instructions for created or updated connections."""
    if not connection:
        return
    if connection.get('aws') and connection['aws'].get('crossAccountRole'):
        obj = {'iamRoleId': connection['aws']['crossAccountRole'].get('iamRoleId'), 'iamUserId': connection['aws']['crossAccountRole'].get('iamUserId'), 'externalId': connection['aws']['crossAccountRole'].get('externalId')}
        if flag_format in ['prettyjson', 'json']:
            _PrintFormattedJsonObject(obj, obj_format=flag_format)
        else:
            print("Please add the following identity to your AWS IAM Role '%s'\nIAM user: '%s'\nExternal Id: '%s'\n" % (connection['aws']['crossAccountRole'].get('iamRoleId'), connection['aws']['crossAccountRole'].get('iamUserId'), connection['aws']['crossAccountRole'].get('externalId')))
    if connection.get('aws') and connection['aws'].get('accessRole'):
        obj = {'iamRoleId': connection['aws']['accessRole'].get('iamRoleId'), 'identity': connection['aws']['accessRole'].get('identity')}
        if flag_format in ['prettyjson', 'json']:
            _PrintFormattedJsonObject(obj, obj_format=flag_format)
        else:
            print("Please add the following identity to your AWS IAM Role '%s'\nIdentity: '%s'\n" % (connection['aws']['accessRole'].get('iamRoleId'), connection['aws']['accessRole'].get('identity')))
    if connection.get('azure') and connection['azure'].get('federatedApplicationClientId'):
        obj = {'federatedApplicationClientId': connection['azure'].get('federatedApplicationClientId'), 'identity': connection['azure'].get('identity')}
        if flag_format in ['prettyjson', 'json']:
            _PrintFormattedJsonObject(obj, obj_format=flag_format)
        else:
            print("Please add the following identity to your Azure application '%s'\nIdentity: '%s'\n" % (connection['azure'].get('federatedApplicationClientId'), connection['azure'].get('identity')))
    elif connection.get('azure'):
        obj = {'clientId': connection['azure'].get('clientId'), 'application': connection['azure'].get('application')}
        if flag_format in ['prettyjson', 'json']:
            _PrintFormattedJsonObject(obj, obj_format=flag_format)
        else:
            print("Please create a Service Principal in your directory for appId: '%s',\nand perform role assignment to app: '%s' to allow BigQuery to access your Azure data. \n" % (connection['azure'].get('clientId'), connection['azure'].get('application')))
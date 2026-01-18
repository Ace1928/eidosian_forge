import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
def _translate_attachments(info):
    attachments = []
    attached_servers = []
    for attachment in info['attachments']:
        attachments.append(attachment['attachment_id'])
        attached_servers.append(attachment['server_id'])
    info.pop('attachments', None)
    info['attachment_ids'] = attachments
    info['attached_servers'] = attached_servers
    return info
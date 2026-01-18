import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class EC2VolumeModification:
    """
    Describes the modification status of an EBS volume.

    If the volume has never been modified, some element values will be null.
    """

    def __init__(self, end_time=None, modification_state=None, original_iops=None, original_size=None, original_volume_type=None, progress=None, start_time=None, status_message=None, target_iops=None, target_size=None, target_volume_type=None, volume_id=None):
        self.end_time = end_time
        self.modification_state = modification_state
        self.original_iops = original_iops
        self.original_size = original_size
        self.original_volume_type = original_volume_type
        self.progress = progress
        self.start_time = start_time
        self.status_message = status_message
        self.target_iops = target_iops
        self.target_size = target_size
        self.target_volume_type = target_volume_type
        self.volume_id = volume_id

    def __repr__(self):
        return '<EC2VolumeModification: end_time=%s, modification_state=%s, original_iops=%s, original_size=%s, original_volume_type=%s, progress=%s, start_time=%s, status_message=%s, target_iops=%s, target_size=%s, target_volume_type=%s, volume_id=%s>' % (self.end_time, self.modification_state, self.original_iops, self.original_size, self.original_volume_type, self.progress, self.start_time, self.status_message, self.target_iops, self.target_size, self.target_volume_type, self.volume_id)
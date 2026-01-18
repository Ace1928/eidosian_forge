import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCEProject(UuidMixin):
    """GCE Project information."""

    def __init__(self, id, name, metadata, quotas, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.metadata = metadata
        self.quotas = quotas
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def set_common_instance_metadata(self, metadata=None, force=False):
        """
        Set common instance metadata for the project. Common uses
        are for setting 'sshKeys', or setting a project-wide
        'startup-script' for all nodes (instances).  Passing in
        ``None`` for the 'metadata' parameter will clear out all common
        instance metadata *except* for 'sshKeys'. If you also want to
        update 'sshKeys', set the 'force' parameter to ``True``.

        :param  metadata: Dictionary of metadata. Can be either a standard
                          python dictionary, or the format expected by
                          GCE (e.g. {'items': [{'key': k1, 'value': v1}, ...}]
        :type   metadata: ``dict`` or ``None``

        :param  force: Force update of 'sshKeys'. If force is ``False`` (the
                       default), existing sshKeys will be retained. Setting
                       force to ``True`` will either replace sshKeys if a new
                       a new value is supplied, or deleted if no new value
                       is supplied.
        :type   force: ``bool``

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_set_common_instance_metadata(metadata=metadata, force=force)

    def set_usage_export_bucket(self, bucket, prefix=None):
        """
        Used to retain Compute Engine resource usage, storing the CSV data in
        a Google Cloud Storage bucket. See the
        `docs <https://cloud.google.com/compute/docs/usage-export>`_ for more
        information. Please ensure you have followed the necessary setup steps
        prior to enabling this feature (e.g. bucket exists, ACLs are in place,
        etc.)

        :param  bucket: Name of the Google Cloud Storage bucket. Specify the
                        name in either 'gs://<bucket_name>' or the full URL
                        'https://storage.googleapis.com/<bucket_name>'.
        :type   bucket: ``str``

        :param  prefix: Optional prefix string for all reports.
        :type   prefix: ``str`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_set_usage_export_bucket(bucket=bucket, prefix=prefix)

    def __repr__(self):
        return '<GCEProject id="{}" name="{}">'.format(self.id, self.name)
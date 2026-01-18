import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def _get_column_value_from_single_row_table(self, table, column):
    """Get the value for the column in the single-row table

        Example table:

        +----------+-------------+----------+----------+
        | address  | cidr        | hostname | host     |
        +----------+-------------+----------+----------+
        | 10.0.0.3 | 10.0.0.0/24 | test     | myhost   |
        +----------+-------------+----------+----------+

        :param table: newline-separated table with |-separated cells
        :param column: name of the column to look for
        :raises: ValueError if the column value is not found
        """
    lines = table.split('\n')
    column_index = -1
    for line in lines:
        if '|' in line:
            if column_index == -1:
                headers = line.split('|')[1:-1]
                for index, header in enumerate(headers):
                    if header.strip() == column:
                        column_index = index
                        break
            else:
                return line.split('|')[1:-1][column_index].strip()
    raise ValueError("Unable to find value for column '%s'." % column)
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
def _get_list_of_values_from_single_column_table(self, table, column):
    """Get the list of values for the column in the single-column table

        Example table:

        +------+
        | Tags |
        +------+
        | tag1 |
        | tag2 |
        +------+

        :param table: newline-separated table with |-separated cells
        :param column: name of the column to look for
        :raises: ValueError if the single column has some other name
        """
    lines = table.split('\n')
    column_name = None
    values = []
    for line in lines:
        if '|' in line:
            if not column_name:
                column_name = line.split('|')[1].strip()
                if column_name != column:
                    raise ValueError('The table has no column %(expected)s but has column %(actual)s.' % {'expected': column, 'actual': column_name})
            else:
                values.append(line.split('|')[1].strip())
    return values
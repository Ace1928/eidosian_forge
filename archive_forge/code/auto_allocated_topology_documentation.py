import argparse
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0
Delete the auto-allocated topology of a given tenant.
from urllib import parse
from oslo_utils import encodeutils
from novaclient import api_versions
from novaclient.tests.unit.fixture_data import base
@staticmethod
def _transform_hypervisor_details(hypervisor):
    """Transform a detailed hypervisor view from 2.53 to 2.88."""
    del hypervisor['current_workload']
    del hypervisor['disk_available_least']
    del hypervisor['free_ram_mb']
    del hypervisor['free_disk_gb']
    del hypervisor['local_gb']
    del hypervisor['local_gb_used']
    del hypervisor['memory_mb']
    del hypervisor['memory_mb_used']
    del hypervisor['running_vms']
    del hypervisor['vcpus']
    del hypervisor['vcpus_used']
    hypervisor['uptime'] = 'fake uptime'
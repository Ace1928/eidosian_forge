from oslo_utils import strutils
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import utils
def _build_body(self, name, ram, vcpus, disk, id, swap, ephemeral, rxtx_factor, is_public):
    return {'flavor': {'name': name, 'ram': ram, 'vcpus': vcpus, 'disk': disk, 'id': id, 'swap': swap, 'OS-FLV-EXT-DATA:ephemeral': ephemeral, 'rxtx_factor': rxtx_factor, 'os-flavor-access:is_public': is_public}}
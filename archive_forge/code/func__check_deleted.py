import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
def _check_deleted(self, ids):
    for _id in ids:
        try:
            utils.get_client(self).template.show(_id)
        except Exception:
            pass
        else:
            return False
    return True
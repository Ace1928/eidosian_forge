from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def _create_new_object(self, object_class, **args):
    new_obj = object_class.new(**args)
    new_obj.Put_()
    return new_obj
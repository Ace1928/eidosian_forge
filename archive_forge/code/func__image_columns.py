from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
def _image_columns(image):
    return image._info.keys()
import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def _translate_starttime(args):
    if args.starttime[0] == '-':
        deltaT = time.time() + int(args.starttime) * 60
        utc = str(datetime.datetime.utcfromtimestamp(deltaT))
        utc = utc.replace(' ', 'T')[:-7] + 'Z'
        args.starttime = utc
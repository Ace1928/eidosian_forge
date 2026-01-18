import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def _wait_for_drive_state_transition(self, drive, state, timeout=DRIVE_TRANSITION_TIMEOUT):
    """
        Wait for a drive to transition to the provided state.

        Note: This function blocks and periodically calls "GET drive" endpoint
        to check if the drive has already transitioned to the desired state.

        :param drive: Drive to wait for.
        :type drive: :class:`.CloudSigmaDrive`

        :param state: Desired drive state.
        :type state: ``str``

        :param timeout: How long to wait for the transition (in seconds) before
                        timing out.
        :type timeout: ``int``

        :return: Drive object.
        :rtype: :class:`.CloudSigmaDrive`
        """
    start_time = time.time()
    while drive.status != state:
        drive = self.ex_get_drive(drive_id=drive.id)
        if drive.status == state:
            break
        current_time = time.time()
        delta = current_time - start_time
        if delta >= timeout:
            msg = 'Timed out while waiting for drive transition (timeout=%s seconds)' % timeout
            raise Exception(msg)
        time.sleep(self.DRIVE_TRANSITION_SLEEP_INTERVAL)
    return drive
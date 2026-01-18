from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def _wait_for_resource_status(cs, resource, expected_status, resource_type='share', status_attr='status', poll_timeout=900, poll_interval=2):
    """Waiter for resource status changes

    :param cs: command shell control
    :param expected_status: a string or a list of strings containing expected
       states to wait for
    :param resource_type: 'share', 'snapshot', 'share_replica', 'share_group',
       or 'share_group_snapshot'
    :param status_attr: 'status', 'task_state', 'access_rules_status' or any
       other status field that is expected to have the "expected_status"
    :param poll_timeout: how long to wait for in seconds
    :param poll_interval: how often to try in seconds
    """
    find_resource = {'share': _find_share, 'snapshot': _find_share_snapshot, 'share_replica': _find_share_replica, 'share_group': _find_share_group, 'share_group_snapshot': _find_share_group_snapshot, 'share_instance': _find_share_instance, 'share_server': _find_share_server, 'share_access_rule': _find_share_access_rule}
    print_resource = {'share': _print_share, 'snapshot': _print_share_snapshot, 'share_replica': _print_share_replica, 'share_group': _print_share_group, 'share_group_snapshot': _print_share_group_snapshot, 'share_instance': _print_share_instance, 'share_access_rule': _print_share_access_rule}
    expected_status = expected_status or ('available',)
    if not isinstance(expected_status, (list, tuple, set)):
        expected_status = (expected_status,)
    time_elapsed = 0
    timeout_message = '%(resource_type)s %(resource)s did not reach %(expected_states)s within %(seconds)d seconds.'
    error_message = '%(resource_type)s %(resource)s has reached a failed state.'
    deleted_message = '%(resource_type)s %(resource)s has been successfully deleted.'
    unmanaged_message = '%(resource_type)s %(resource)s has been successfully unmanaged.'
    message_payload = {'resource_type': resource_type.capitalize(), 'resource': resource.id}
    not_found_regex = 'no .* exists'
    while True:
        if time_elapsed > poll_timeout:
            print_resource[resource_type](cs, resource)
            message_payload.update({'expected_states': expected_status, 'seconds': poll_timeout})
            raise exceptions.TimeoutException(message=timeout_message % message_payload)
        try:
            resource = find_resource[resource_type](cs, resource.id)
        except exceptions.CommandError as e:
            if re.search(not_found_regex, str(e), flags=re.IGNORECASE):
                if 'deleted' in expected_status:
                    print(deleted_message % message_payload)
                    break
                if 'unmanaged' in expected_status:
                    print(unmanaged_message % message_payload)
                    break
            else:
                raise e
        if getattr(resource, status_attr) in expected_status:
            break
        elif 'error' in getattr(resource, status_attr):
            print_resource[resource_type](cs, resource)
            raise exceptions.ResourceInErrorState(message=error_message % message_payload)
        time.sleep(poll_interval)
        time_elapsed += poll_interval
    return resource
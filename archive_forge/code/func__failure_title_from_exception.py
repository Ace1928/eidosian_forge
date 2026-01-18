from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def _failure_title_from_exception(exception):
    """Compose human-readable title for module error title.

    Title will be based on status codes if they has been provided.
    :type exception:  exceptions.GeneralPubNubError
    :param exception: Reference on exception for which title should be
                      composed.

    :rtype:  str
    :return: Reference on error tile which should be shown on module
             failure.
    """
    title = 'General REST API access error.'
    if exception.code == exceptions.PN_AUTHORIZATION_MISSING_CREDENTIALS:
        title = 'Authorization error: missing credentials.'
    elif exception.code == exceptions.PN_AUTHORIZATION_WRONG_CREDENTIALS:
        title = 'Authorization error: wrong credentials.'
    elif exception.code == exceptions.PN_USER_INSUFFICIENT_RIGHTS:
        title = 'API access error: insufficient access rights.'
    elif exception.code == exceptions.PN_API_ACCESS_TOKEN_EXPIRED:
        title = 'API access error: time token expired.'
    elif exception.code == exceptions.PN_KEYSET_BLOCK_EXISTS:
        title = 'Block create did fail: block with same name already exists).'
    elif exception.code == exceptions.PN_KEYSET_BLOCKS_FETCH_DID_FAIL:
        title = 'Unable fetch list of blocks for keyset.'
    elif exception.code == exceptions.PN_BLOCK_CREATE_DID_FAIL:
        title = 'Block creation did fail.'
    elif exception.code == exceptions.PN_BLOCK_UPDATE_DID_FAIL:
        title = 'Block update did fail.'
    elif exception.code == exceptions.PN_BLOCK_REMOVE_DID_FAIL:
        title = 'Block removal did fail.'
    elif exception.code == exceptions.PN_BLOCK_START_STOP_DID_FAIL:
        title = 'Block start/stop did fail.'
    elif exception.code == exceptions.PN_EVENT_HANDLER_MISSING_FIELDS:
        title = 'Event handler creation did fail: missing fields.'
    elif exception.code == exceptions.PN_BLOCK_EVENT_HANDLER_EXISTS:
        title = 'Event handler creation did fail: missing fields.'
    elif exception.code == exceptions.PN_EVENT_HANDLER_CREATE_DID_FAIL:
        title = 'Event handler creation did fail.'
    elif exception.code == exceptions.PN_EVENT_HANDLER_UPDATE_DID_FAIL:
        title = 'Event handler update did fail.'
    elif exception.code == exceptions.PN_EVENT_HANDLER_REMOVE_DID_FAIL:
        title = 'Event handler removal did fail.'
    return title
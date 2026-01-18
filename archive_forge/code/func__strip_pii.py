from __future__ import absolute_import
import copy
from sentry_sdk import Hub
from sentry_sdk.consts import SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _strip_pii(command):
    for key in command:
        is_safe_field = key in SAFE_COMMAND_ATTRIBUTES
        if is_safe_field:
            continue
        update_db_command = key == 'update' and 'findAndModify' not in command
        if update_db_command:
            continue
        is_document = key == 'documents'
        if is_document:
            for doc in command[key]:
                for doc_key in doc:
                    doc[doc_key] = '%s'
            continue
        is_dict_field = key in ['filter', 'query', 'update']
        if is_dict_field:
            for item_key in command[key]:
                command[key][item_key] = '%s'
            continue
        is_pipeline_field = key == 'pipeline'
        if is_pipeline_field:
            for pipeline in command[key]:
                for match_key in pipeline['$match'] if '$match' in pipeline else []:
                    pipeline['$match'][match_key] = '%s'
            continue
        command[key] = '%s'
    return command
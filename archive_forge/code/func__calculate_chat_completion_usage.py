from sentry_sdk import consts
from sentry_sdk._types import TYPE_CHECKING
import sentry_sdk
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions, event_from_exception
def _calculate_chat_completion_usage(messages, response, span, streaming_message_responses=None):
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0
    if hasattr(response, 'usage'):
        if hasattr(response.usage, 'completion_tokens') and isinstance(response.usage.completion_tokens, int):
            completion_tokens = response.usage.completion_tokens
        if hasattr(response.usage, 'prompt_tokens') and isinstance(response.usage.prompt_tokens, int):
            prompt_tokens = response.usage.prompt_tokens
        if hasattr(response.usage, 'total_tokens') and isinstance(response.usage.total_tokens, int):
            total_tokens = response.usage.total_tokens
    if prompt_tokens == 0:
        for message in messages:
            if 'content' in message:
                prompt_tokens += count_tokens(message['content'])
    if completion_tokens == 0:
        if streaming_message_responses is not None:
            for message in streaming_message_responses:
                completion_tokens += count_tokens(message)
        elif hasattr(response, 'choices'):
            for choice in response.choices:
                if hasattr(choice, 'message'):
                    completion_tokens += count_tokens(choice.message)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    if completion_tokens != 0:
        set_data_normalized(span, COMPLETION_TOKENS_USED, completion_tokens)
    if prompt_tokens != 0:
        set_data_normalized(span, PROMPT_TOKENS_USED, prompt_tokens)
    if total_tokens != 0:
        set_data_normalized(span, TOTAL_TOKENS_USED, total_tokens)
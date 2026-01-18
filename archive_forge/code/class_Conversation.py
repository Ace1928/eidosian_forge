import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    [`ConversationalPipeline`]. The conversation contains several utility functions to manage the addition of new user
    inputs and generated model responses.

    Arguments:
        messages (Union[str, List[Dict[str, str]]], *optional*):
            The initial messages to start the conversation, either a string, or a list of dicts containing "role" and
            "content" keys. If a string is passed, it is interpreted as a single message with the "user" role.
        conversation_id (`uuid.UUID`, *optional*):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")
    conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
    conversation.add_message({"role": "user", "content": "Is it good?"})
    ```"""

    def __init__(self, messages: Union[str, List[Dict[str, str]]]=None, conversation_id: uuid.UUID=None, **deprecated_kwargs):
        if not conversation_id:
            conversation_id = uuid.uuid4()
        if messages is None:
            text = deprecated_kwargs.pop('text', None)
            if text is not None:
                messages = [{'role': 'user', 'content': text}]
            else:
                messages = []
        elif isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        self._num_processed_user_inputs = 0
        generated_responses = deprecated_kwargs.pop('generated_responses', None)
        past_user_inputs = deprecated_kwargs.pop('past_user_inputs', None)
        if generated_responses is not None and past_user_inputs is None:
            raise ValueError('generated_responses cannot be passed without past_user_inputs!')
        if past_user_inputs is not None:
            legacy_messages = []
            if generated_responses is None:
                generated_responses = []
            for i in range(max([len(past_user_inputs), len(generated_responses)])):
                if i < len(past_user_inputs):
                    legacy_messages.append({'role': 'user', 'content': past_user_inputs[i]})
                if i < len(generated_responses):
                    legacy_messages.append({'role': 'assistant', 'content': generated_responses[i]})
            messages = legacy_messages + messages
        self.uuid = conversation_id
        self.messages = messages

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.uuid == other.uuid or self.messages == other.messages

    def add_message(self, message: Dict[str, str]):
        if not set(message.keys()) == {'role', 'content'}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")
        if message['role'] not in ('user', 'assistant', 'system'):
            raise ValueError("Only 'user', 'assistant' and 'system' roles are supported for now!")
        self.messages.append(message)

    def add_user_input(self, text: str, overwrite: bool=False):
        """
        Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must
        alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend
        just using `add_message` with role "user" instead.
        """
        if len(self) > 0 and self[-1]['role'] == 'user':
            if overwrite:
                logger.warning(f'User input added while unprocessed input was existing: "{self[-1]['content']}" was overwritten with: "{text}".')
                self[-1]['content'] = text
            else:
                logger.warning(f'User input added while unprocessed input was existing: "{self[-1]['content']}" new input ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input')
        else:
            self.messages.append({'role': 'user', 'content': text})

    def append_response(self, response: str):
        """
        This is a legacy method. We recommend just using `add_message` with an appropriate role instead.
        """
        self.messages.append({'role': 'assistant', 'content': response})

    def mark_processed(self):
        """
        This is a legacy method, as the Conversation no longer distinguishes between processed and unprocessed user
        input. We set a counter here to keep behaviour mostly backward-compatible, but in general you should just read
        the messages directly when writing new code.
        """
        self._num_processed_user_inputs = len(self._user_messages)

    def __iter__(self):
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        return self.messages[item]

    def __setitem__(self, key, value):
        self.messages[key] = value

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Returns:
            `str`:

        Example:
            Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user: Going to the movies tonight - any suggestions?
            bot: The Big Lebowski
        """
        output = f'Conversation id: {self.uuid}\n'
        for message in self.messages:
            output += f'{message['role']}: {message['content']}\n'
        return output

    def iter_texts(self):
        for message in self.messages:
            yield (message['role'] == 'user', message['content'])

    @property
    def _user_messages(self):
        return [message['content'] for message in self.messages if message['role'] == 'user']

    @property
    def past_user_inputs(self):
        if not self._user_messages:
            return []
        if self.messages[-1]['role'] != 'user' or self._num_processed_user_inputs == len(self._user_messages):
            return self._user_messages
        return self._user_messages[:-1]

    @property
    def generated_responses(self):
        return [message['content'] for message in self.messages if message['role'] == 'assistant']

    @property
    def new_user_input(self):
        return self._user_messages[-1]
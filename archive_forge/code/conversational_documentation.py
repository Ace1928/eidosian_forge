import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args

        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a [`Conversation`] or a list of [`Conversation`]):
                Conversation to generate responses for. Inputs can also be passed as a list of dictionaries with `role`
                and `content` keys - in this case, they will be converted to `Conversation` objects automatically.
                Multiple conversations in either format may be passed as a list.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            [`Conversation`] or a list of [`Conversation`]: Conversation(s) with updated generated responses for those
            containing a new user input.
        
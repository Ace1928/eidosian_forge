from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast
from streamlit import runtime
from streamlit.elements.form import is_in_form
from streamlit.elements.image import AtomicImage, WidthBehaviour, image_to_url
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.proto.Common_pb2 import StringTriggerValue as StringTriggerValueProto
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import is_emoji
from streamlit.type_util import Key, to_key
def _process_avatar_input(avatar: str | AtomicImage | None, delta_path: str) -> tuple[BlockProto.ChatMessage.AvatarType.ValueType, str]:
    """Detects the avatar type and prepares the avatar data for the frontend.

    Parameters
    ----------
    avatar :
        The avatar that was provided by the user.
    delta_path : str
        The delta path is used as media ID when a local image is served via the media
        file manager.

    Returns
    -------
    Tuple[AvatarType, str]
        The detected avatar type and the prepared avatar data.
    """
    AvatarType = BlockProto.ChatMessage.AvatarType
    if avatar is None:
        return (AvatarType.ICON, '')
    elif isinstance(avatar, str) and avatar in {item.value for item in PresetNames}:
        return (AvatarType.ICON, 'assistant' if avatar in [PresetNames.AI, PresetNames.ASSISTANT] else 'user')
    elif isinstance(avatar, str) and is_emoji(avatar):
        return (AvatarType.EMOJI, avatar)
    else:
        try:
            return (AvatarType.IMAGE, image_to_url(avatar, width=WidthBehaviour.ORIGINAL, clamp=False, channels='RGB', output_format='auto', image_id=delta_path))
        except Exception as ex:
            raise StreamlitAPIException('Failed to load the provided avatar value as an image.') from ex
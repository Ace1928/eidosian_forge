from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class CloseStreamMessage(BaseModel):
    msg: Literal[ServerMessage.close_stream] = ServerMessage.close_stream
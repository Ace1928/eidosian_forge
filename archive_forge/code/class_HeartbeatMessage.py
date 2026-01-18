from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class HeartbeatMessage(BaseModel):
    msg: Literal[ServerMessage.heartbeat] = ServerMessage.heartbeat
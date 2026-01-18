from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class ProcessCompletedMessage(BaseMessage):
    msg: Literal[ServerMessage.process_completed] = ServerMessage.process_completed
    output: dict
    success: bool
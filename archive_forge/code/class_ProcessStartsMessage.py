from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class ProcessStartsMessage(BaseMessage):
    msg: Literal[ServerMessage.process_starts] = ServerMessage.process_starts
    eta: Optional[float] = None
from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class UnexpectedErrorMessage(BaseModel):
    msg: Literal[ServerMessage.unexpected_error] = ServerMessage.unexpected_error
    message: str
    success: Literal[False] = False
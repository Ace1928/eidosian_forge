from typing import Any, Dict, List, Optional
from jupyter_client.client import KernelClient
from nbformat.v4 import output_from_msg
from .jsonutil import json_clean
def _publish_msg(self, msg_type: str, data: Optional[Dict]=None, metadata: Optional[Dict]=None, buffers: Optional[List]=None, **keys: Any) -> None:
    """Helper for sending a comm message on IOPub"""
    data = {} if data is None else data
    metadata = {} if metadata is None else metadata
    content = json_clean(dict(data=data, comm_id=self.comm_id, **keys))
    msg = self.kernel_client.session.msg(msg_type, content=content, parent=self.parent_header, metadata=metadata)
    self.kernel_client.shell_channel.send(msg)
import time
from typing import Any, Dict, List, Optional, cast
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
def _send_to_infino(self, key: str, value: Any, is_ts: bool=True) -> None:
    """Send the key-value to Infino.

        Parameters:
        key (str): the key to send to Infino.
        value (Any): the value to send to Infino.
        is_ts (bool): if True, the value is part of a time series, else it
                      is sent as a log message.
        """
    payload = {'date': int(time.time()), key: value, 'labels': {'model_id': self.model_id, 'model_version': self.model_version}}
    if self.verbose:
        print(f'Tracking {key} with Infino: {payload}')
    if is_ts:
        self.client.append_ts(payload)
    else:
        self.client.append_log(payload)
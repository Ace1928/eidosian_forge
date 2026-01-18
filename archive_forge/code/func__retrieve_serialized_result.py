from typing import List, Optional
import time
import requests
import cirq
import cirq_pasqal
def _retrieve_serialized_result(self, task_id: str) -> str:
    """Retrieves the results from the remote Pasqal device
        Args:
            task_id: id of the current task.
        Returns:
            json representation of the results
        """
    url = f'{self.remote_host}/get-result/{task_id}'
    while True:
        response = requests.get(url, headers=self._authorization_header, verify=False)
        response.raise_for_status()
        result = response.text
        if result:
            return result
        time.sleep(1.0)
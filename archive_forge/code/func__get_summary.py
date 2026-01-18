import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
def _get_summary(self) -> Dict:
    try:
        response = self.client.get_propertygraph_summary()
    except Exception as e:
        raise NeptuneQueryException({'message': 'Summary API is not available for this instance of Neptune,ensure the engine version is >=1.2.1.0', 'details': str(e)})
    try:
        summary = response['payload']['graphSummary']
    except Exception:
        raise NeptuneQueryException({'message': 'Summary API did not return a valid response.', 'details': response.content.decode()})
    else:
        return summary
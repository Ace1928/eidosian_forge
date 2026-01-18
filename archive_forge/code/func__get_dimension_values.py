import json
import logging
import time
from typing import Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_dimension_values(self, dimension_name: str) -> List[str]:
    """Makes a call to Cube's REST API load endpoint to retrieve
        values for dimensions.

        These values can be used to achieve a more accurate filtering.
        """
    logger.info('Loading dimension values for: {dimension_name}...')
    headers = {'Content-Type': 'application/json', 'Authorization': self.cube_api_token}
    query = {'query': {'dimensions': [dimension_name], 'limit': self.dimension_values_limit}}
    retries = 0
    while retries < self.dimension_values_max_retries:
        response = requests.request('POST', f'{self.cube_api_url}/load', headers=headers, data=json.dumps(query))
        if response.status_code == 200:
            response_data = response.json()
            if 'error' in response_data and response_data['error'] == 'Continue wait':
                logger.info('Retrying...')
                retries += 1
                time.sleep(self.dimension_values_retry_delay)
                continue
            else:
                dimension_values = [item[dimension_name] for item in response_data['data']]
                return dimension_values
        else:
            logger.error('Request failed with status code:', response.status_code)
            break
    if retries == self.dimension_values_max_retries:
        logger.info('Maximum retries reached.')
    return []
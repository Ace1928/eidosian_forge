import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
class CohereRequestResponseResolver:
    """Class to resolve the request/response from the Cohere API and convert it to a dictionary that can be logged."""

    def __call__(self, args: Sequence[Any], kwargs: Dict[str, Any], response: Response, start_time: float, time_elapsed: float) -> Optional[Dict[str, Any]]:
        """Process the response from the Cohere API and convert it to a dictionary that can be logged.

        :param args: The arguments of the original function.
        :param kwargs: The keyword arguments of the original function.
        :param response: The response from the Cohere API.
        :param start_time: The start time of the request.
        :param time_elapsed: The time elapsed for the request.
        :return: A dictionary containing the parsed response and timing information.
        """
        try:
            response_type = str(type(response)).split("'")[1].split('.')[-1]
            parsed_response = None
            if response_type == 'Generations':
                parsed_response = self._resolve_generate_response(response)
                table_column_order = ['start_time', 'query_id', 'model', 'prompt', 'text', 'token_likelihoods', 'likelihood', 'time_elapsed_(seconds)', 'end_time']
                default_model = 'command'
            elif response_type == 'Chat':
                parsed_response = self._resolve_chat_response(response)
                table_column_order = ['start_time', 'query_id', 'model', 'conversation_id', 'response_id', 'query', 'text', 'prompt', 'preamble', 'chat_history', 'chatlog', 'time_elapsed_(seconds)', 'end_time']
                default_model = 'command'
            elif response_type == 'Classifications':
                parsed_response = self._resolve_classify_response(response)
                kwargs = self._resolve_classify_kwargs(kwargs)
                table_column_order = ['start_time', 'query_id', 'model', 'id', 'input', 'prediction', 'confidence', 'time_elapsed_(seconds)', 'end_time']
                default_model = 'embed-english-v2.0'
            elif response_type == 'SummarizeResponse':
                parsed_response = self._resolve_summarize_response(response)
                table_column_order = ['start_time', 'query_id', 'model', 'response_id', 'text', 'additional_command', 'summary', 'time_elapsed_(seconds)', 'end_time', 'length', 'format']
                default_model = 'summarize-xlarge'
            elif response_type == 'Reranking':
                parsed_response = self._resolve_rerank_response(response)
                table_column_order = ['start_time', 'query_id', 'model', 'id', 'query', 'top_n', 'document-text', 'relevance_score', 'index', 'time_elapsed_(seconds)', 'end_time']
                default_model = 'rerank-english-v2.0'
            else:
                logger.info(f'Unsupported Cohere response object: {response}')
            return self._resolve(args, kwargs, parsed_response, start_time, time_elapsed, response_type, table_column_order, default_model)
        except Exception as e:
            logger.warning(f'Failed to resolve request/response: {e}')
        return None

    def _resolve_generate_response(self, response: Response) -> List[Dict[str, Any]]:
        return_list = []
        for _response in response:
            _response_dict = _response._visualize_helper()
            try:
                _response_dict['token_likelihoods'] = wandb.Html(_response_dict['token_likelihoods'])
            except (KeyError, ValueError):
                pass
            return_list.append(_response_dict)
        return return_list

    def _resolve_chat_response(self, response: Response) -> List[Dict[str, Any]]:
        return [subset_dict(response.__dict__, ['response_id', 'generation_id', 'query', 'text', 'conversation_id', 'prompt', 'chatlog', 'preamble'])]

    def _resolve_classify_response(self, response: Response) -> List[Dict[str, Any]]:
        return [flatten_dict(_response.__dict__) for _response in response]

    def _resolve_classify_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        example_texts = []
        example_labels = []
        for example in kwargs['examples']:
            example_texts.append(example.text)
            example_labels.append(example.label)
        kwargs.pop('examples')
        kwargs['example_texts'] = example_texts
        kwargs['example_labels'] = example_labels
        return kwargs

    def _resolve_summarize_response(self, response: Response) -> List[Dict[str, Any]]:
        return [{'response_id': response.id, 'summary': response.summary}]

    def _resolve_rerank_response(self, response: Response) -> List[Dict[str, Any]]:
        flattened_response_dicts = [flatten_dict(_response.__dict__) for _response in response]
        return_dict = collect_common_keys(flattened_response_dicts)
        return_dict['id'] = response.id
        return [return_dict]

    def _resolve(self, args: Sequence[Any], kwargs: Dict[str, Any], parsed_response: List[Dict[str, Any]], start_time: float, time_elapsed: float, response_type: str, table_column_order: List[str], default_model: str) -> Dict[str, Any]:
        """Convert a list of dictionaries to a pair of column names and corresponding values, with the option to order specific dictionaries.

        :param args: The arguments passed to the API client.
        :param kwargs: The keyword arguments passed to the API client.
        :param parsed_response: The parsed response from the API.
        :param start_time: The start time of the API request.
        :param time_elapsed: The time elapsed during the API request.
        :param response_type: The type of the API response.
        :param table_column_order: The desired order of columns in the resulting table.
        :param default_model: The default model to use if not specified in the response.
        :return: A dictionary containing the formatted response.
        """
        query_id = generate_id(length=16)
        parsed_args = subset_dict(args[0].__dict__, ['api_version', 'batch_size', 'max_retries', 'num_workers', 'timeout'])
        start_time_dt = datetime.fromtimestamp(start_time)
        end_time_dt = datetime.fromtimestamp(start_time + time_elapsed)
        timings = {'start_time': start_time_dt, 'end_time': end_time_dt, 'time_elapsed_(seconds)': time_elapsed}
        packed_data = []
        for _parsed_response in parsed_response:
            _packed_dict = {'query_id': query_id, **kwargs, **_parsed_response, **timings, **parsed_args}
            if 'model' not in _packed_dict:
                _packed_dict['model'] = default_model
            packed_data.append(_packed_dict)
        columns, data = reorder_and_convert_dict_list_to_table(packed_data, table_column_order)
        request_response_table = wandb.Table(data=data, columns=columns)
        return {f'{response_type}': request_response_table}
import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
class PageIterator:
    """An iterable object to paginate API results.
    Please note it is NOT a python iterator.
    Use ``iter`` to wrap this as a generator.
    """

    def __init__(self, method, input_token, output_token, more_results, result_keys, non_aggregate_keys, limit_key, max_items, starting_token, page_size, op_kwargs):
        self._method = method
        self._input_token = input_token
        self._output_token = output_token
        self._more_results = more_results
        self._result_keys = result_keys
        self._max_items = max_items
        self._limit_key = limit_key
        self._starting_token = starting_token
        self._page_size = page_size
        self._op_kwargs = op_kwargs
        self._resume_token = None
        self._non_aggregate_key_exprs = non_aggregate_keys
        self._non_aggregate_part = {}
        self._token_encoder = TokenEncoder()
        self._token_decoder = TokenDecoder()

    @property
    def result_keys(self):
        return self._result_keys

    @property
    def resume_token(self):
        """Token to specify to resume pagination."""
        return self._resume_token

    @resume_token.setter
    def resume_token(self, value):
        if not isinstance(value, dict):
            raise ValueError('Bad starting token: %s' % value)
        if 'boto_truncate_amount' in value:
            token_keys = sorted(self._input_token + ['boto_truncate_amount'])
        else:
            token_keys = sorted(self._input_token)
        dict_keys = sorted(value.keys())
        if token_keys == dict_keys:
            self._resume_token = self._token_encoder.encode(value)
        else:
            raise ValueError('Bad starting token: %s' % value)

    @property
    def non_aggregate_part(self):
        return self._non_aggregate_part

    def __iter__(self):
        current_kwargs = self._op_kwargs
        previous_next_token = None
        next_token = {key: None for key in self._input_token}
        if self._starting_token is not None:
            next_token = self._parse_starting_token()[0]
        total_items = 0
        first_request = True
        primary_result_key = self.result_keys[0]
        starting_truncation = 0
        self._inject_starting_params(current_kwargs)
        while True:
            response = self._make_request(current_kwargs)
            parsed = self._extract_parsed_response(response)
            if first_request:
                if self._starting_token is not None:
                    starting_truncation = self._handle_first_request(parsed, primary_result_key, starting_truncation)
                first_request = False
                self._record_non_aggregate_key_values(parsed)
            else:
                starting_truncation = 0
            current_response = primary_result_key.search(parsed)
            if current_response is None:
                current_response = []
            num_current_response = len(current_response)
            truncate_amount = 0
            if self._max_items is not None:
                truncate_amount = total_items + num_current_response - self._max_items
            if truncate_amount > 0:
                self._truncate_response(parsed, primary_result_key, truncate_amount, starting_truncation, next_token)
                yield response
                break
            else:
                yield response
                total_items += num_current_response
                next_token = self._get_next_token(parsed)
                if all((t is None for t in next_token.values())):
                    break
                if self._max_items is not None and total_items == self._max_items:
                    self.resume_token = next_token
                    break
                if previous_next_token is not None and previous_next_token == next_token:
                    message = f'The same next token was received twice: {next_token}'
                    raise PaginationError(message=message)
                self._inject_token_into_kwargs(current_kwargs, next_token)
                previous_next_token = next_token

    def search(self, expression):
        """Applies a JMESPath expression to a paginator

        Each page of results is searched using the provided JMESPath
        expression. If the result is not a list, it is yielded
        directly. If the result is a list, each element in the result
        is yielded individually (essentially implementing a flatmap in
        which the JMESPath search is the mapping function).

        :type expression: str
        :param expression: JMESPath expression to apply to each page.

        :return: Returns an iterator that yields the individual
            elements of applying a JMESPath expression to each page of
            results.
        """
        compiled = jmespath.compile(expression)
        for page in self:
            results = compiled.search(page)
            if isinstance(results, list):
                yield from results
            else:
                yield results

    def _make_request(self, current_kwargs):
        return self._method(**current_kwargs)

    def _extract_parsed_response(self, response):
        return response

    def _record_non_aggregate_key_values(self, response):
        non_aggregate_keys = {}
        for expression in self._non_aggregate_key_exprs:
            result = expression.search(response)
            set_value_from_jmespath(non_aggregate_keys, expression.expression, result)
        self._non_aggregate_part = non_aggregate_keys

    def _inject_starting_params(self, op_kwargs):
        if self._starting_token is not None:
            next_token = self._parse_starting_token()[0]
            self._inject_token_into_kwargs(op_kwargs, next_token)
        if self._page_size is not None:
            op_kwargs[self._limit_key] = self._page_size

    def _inject_token_into_kwargs(self, op_kwargs, next_token):
        for name, token in next_token.items():
            if token is not None and token != 'None':
                op_kwargs[name] = token
            elif name in op_kwargs:
                del op_kwargs[name]

    def _handle_first_request(self, parsed, primary_result_key, starting_truncation):
        starting_truncation = self._parse_starting_token()[1]
        all_data = primary_result_key.search(parsed)
        if isinstance(all_data, (list, str)):
            data = all_data[starting_truncation:]
        else:
            data = None
        set_value_from_jmespath(parsed, primary_result_key.expression, data)
        for token in self.result_keys:
            if token == primary_result_key:
                continue
            sample = token.search(parsed)
            if isinstance(sample, list):
                empty_value = []
            elif isinstance(sample, str):
                empty_value = ''
            elif isinstance(sample, (int, float)):
                empty_value = 0
            else:
                empty_value = None
            set_value_from_jmespath(parsed, token.expression, empty_value)
        return starting_truncation

    def _truncate_response(self, parsed, primary_result_key, truncate_amount, starting_truncation, next_token):
        original = primary_result_key.search(parsed)
        if original is None:
            original = []
        amount_to_keep = len(original) - truncate_amount
        truncated = original[:amount_to_keep]
        set_value_from_jmespath(parsed, primary_result_key.expression, truncated)
        next_token['boto_truncate_amount'] = amount_to_keep + starting_truncation
        self.resume_token = next_token

    def _get_next_token(self, parsed):
        if self._more_results is not None:
            if not self._more_results.search(parsed):
                return {}
        next_tokens = {}
        for output_token, input_key in zip(self._output_token, self._input_token):
            next_token = output_token.search(parsed)
            if next_token:
                next_tokens[input_key] = next_token
            else:
                next_tokens[input_key] = None
        return next_tokens

    def result_key_iters(self):
        teed_results = tee(self, len(self.result_keys))
        return [ResultKeyIterator(i, result_key) for i, result_key in zip(teed_results, self.result_keys)]

    def build_full_result(self):
        complete_result = {}
        for response in self:
            page = response
            if isinstance(response, tuple) and len(response) == 2:
                page = response[1]
            for result_expression in self.result_keys:
                result_value = result_expression.search(page)
                if result_value is None:
                    continue
                existing_value = result_expression.search(complete_result)
                if existing_value is None:
                    set_value_from_jmespath(complete_result, result_expression.expression, result_value)
                    continue
                if isinstance(result_value, list):
                    existing_value.extend(result_value)
                elif isinstance(result_value, (int, float, str)):
                    set_value_from_jmespath(complete_result, result_expression.expression, existing_value + result_value)
        merge_dicts(complete_result, self.non_aggregate_part)
        if self.resume_token is not None:
            complete_result['NextToken'] = self.resume_token
        return complete_result

    def _parse_starting_token(self):
        if self._starting_token is None:
            return None
        next_token = self._starting_token
        try:
            next_token = self._token_decoder.decode(next_token)
            index = 0
            if 'boto_truncate_amount' in next_token:
                index = next_token.get('boto_truncate_amount')
                del next_token['boto_truncate_amount']
        except (ValueError, TypeError):
            next_token, index = self._parse_starting_token_deprecated()
        return (next_token, index)

    def _parse_starting_token_deprecated(self):
        """
        This handles parsing of old style starting tokens, and attempts to
        coerce them into the new style.
        """
        log.debug('Attempting to fall back to old starting token parser. For token: %s' % self._starting_token)
        if self._starting_token is None:
            return None
        parts = self._starting_token.split('___')
        next_token = []
        index = 0
        if len(parts) == len(self._input_token) + 1:
            try:
                index = int(parts.pop())
            except ValueError:
                parts = [self._starting_token]
        for part in parts:
            if part == 'None':
                next_token.append(None)
            else:
                next_token.append(part)
        return (self._convert_deprecated_starting_token(next_token), index)

    def _convert_deprecated_starting_token(self, deprecated_token):
        """
        This attempts to convert a deprecated starting token into the new
        style.
        """
        len_deprecated_token = len(deprecated_token)
        len_input_token = len(self._input_token)
        if len_deprecated_token > len_input_token:
            raise ValueError('Bad starting token: %s' % self._starting_token)
        elif len_deprecated_token < len_input_token:
            log.debug('Old format starting token does not contain all input tokens. Setting the rest, in order, as None.')
            for i in range(len_input_token - len_deprecated_token):
                deprecated_token.append(None)
        return dict(zip(self._input_token, deprecated_token))
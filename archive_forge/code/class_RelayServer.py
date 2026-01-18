import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
class RelayServer:

    def __init__(self, base_url: str, inject: Optional[List[InjectedResponse]]=None, control: Optional[RelayControlProtocol]=None, verbose: bool=False) -> None:
        self.relay_control = control
        self.app = flask.Flask(__name__)
        self.app.logger.setLevel(logging.INFO)
        self.app.register_error_handler(DeliberateHTTPError, self.handle_http_exception)
        self.app.add_url_rule(rule='/graphql', endpoint='graphql', view_func=self.graphql, methods=['POST'])
        self.app.add_url_rule(rule='/files/<path:path>', endpoint='files', view_func=self.file_stream, methods=['POST'])
        self.app.add_url_rule(rule='/storage', endpoint='storage', view_func=self.storage, methods=['PUT', 'GET'])
        self.app.add_url_rule(rule='/storage/<path:path>', endpoint='storage_file', view_func=self.storage_file, methods=['PUT', 'GET'])
        if control:
            self.app.add_url_rule(rule='/_control', endpoint='_control', view_func=self.control, methods=['POST'])
        self.port = self._get_free_port()
        self.base_url = urllib.parse.urlparse(base_url)
        self.session = requests.Session()
        self.relay_url = f'http://127.0.0.1:{self.port}'
        self.resolver = QueryResolver()
        self.context = Context()
        self.inject = inject or []
        self.verbose = verbose

    @staticmethod
    def handle_http_exception(e):
        response = e.get_response()
        return response

    @staticmethod
    def _get_free_port() -> int:
        sock = socket.socket()
        sock.bind(('', 0))
        _, port = sock.getsockname()
        return port

    def start(self) -> None:
        relay_server_thread = threading.Thread(target=self.app.run, kwargs={'port': self.port}, daemon=True)
        relay_server_thread.start()

    def after_request_fn(self, response: 'requests.Response') -> 'requests.Response':
        print(flask.request)
        print(flask.request.get_json())
        print(response)
        print(response.json())
        return response

    def relay(self, request: 'flask.Request') -> Union['responses.Response', 'requests.Response']:
        url = urllib.parse.urlparse(request.url)._replace(netloc=self.base_url.netloc, scheme=self.base_url.scheme).geturl()
        headers = {key: value for key, value in request.headers if key != 'Host'}
        prepared_relayed_request = requests.Request(method=request.method, url=url, headers=headers, data=request.get_data(), json=request.get_json()).prepare()
        if self.verbose:
            print('*****************')
            print('RELAY REQUEST:')
            print(prepared_relayed_request.url)
            print(prepared_relayed_request.method)
            print(prepared_relayed_request.headers)
            print(prepared_relayed_request.body)
            print('*****************')
        for injected_response in self.inject:
            should_apply = injected_response.application_pattern.should_apply()
            if injected_response == prepared_relayed_request:
                if self.verbose:
                    print('*****************')
                    print('INJECTING RESPONSE:')
                    print(injected_response.to_dict())
                    print('*****************')
                injected_response.application_pattern.next()
                if should_apply:
                    with responses.RequestsMock() as mocked_responses:
                        mocked_responses.add(**injected_response.to_dict())
                        relayed_response = self.session.send(prepared_relayed_request)
                        return relayed_response
        relayed_response = self.session.send(prepared_relayed_request)
        return relayed_response

    def snoop_context(self, request: 'flask.Request', response: 'requests.Response', time_elapsed: float, **kwargs: Any) -> None:
        request_data = request.get_json()
        response_data = response.json() or {}
        if self.relay_control:
            self.relay_control.process(request)
        raw_data: RawRequestResponse = {'url': request.url, 'request': request_data, 'response': response_data, 'time_elapsed': time_elapsed}
        self.context.raw_data.append(raw_data)
        try:
            snooped_context = self.resolver.resolve(request_data, response_data, **kwargs)
        except Exception as e:
            print('Failed to resolve context: ', e)
            traceback.print_exc()
            snooped_context = None
        if snooped_context is not None:
            self.context.upsert(snooped_context)
        return None

    def graphql(self) -> Mapping[str, str]:
        request = flask.request
        with Timer() as timer:
            relayed_response = self.relay(request)
        if self.verbose:
            print('*****************')
            print('GRAPHQL REQUEST:')
            print(request.get_json())
            print('GRAPHQL RESPONSE:')
            print(relayed_response.status_code, relayed_response.json())
            print('*****************')
        self.snoop_context(request, relayed_response, timer.elapsed)
        if self.verbose:
            print('*****************')
            print('SNOOPED CONTEXT:')
            print(self.context.entries)
            print(len(self.context.raw_data))
            print('*****************')
        return relayed_response.json()

    def file_stream(self, path) -> Mapping[str, str]:
        request = flask.request
        with Timer() as timer:
            relayed_response = self.relay(request)
        if self.verbose:
            print('*****************')
            print('FILE STREAM REQUEST:')
            print('********PATH*********')
            print(path)
            print('********ENDPATH*********')
            print(request.get_json())
            print('FILE STREAM RESPONSE:')
            print(relayed_response)
            print(relayed_response.status_code, relayed_response.json())
            print('*****************')
        self.snoop_context(request, relayed_response, timer.elapsed, path=path)
        return relayed_response.json()

    def storage(self) -> Mapping[str, str]:
        request = flask.request
        with Timer() as timer:
            relayed_response = self.relay(request)
        if self.verbose:
            print('*****************')
            print('STORAGE REQUEST:')
            print(request.get_json())
            print('STORAGE RESPONSE:')
            print(relayed_response.status_code, relayed_response.json())
            print('*****************')
        self.snoop_context(request, relayed_response, timer.elapsed)
        return relayed_response.json()

    def storage_file(self, path) -> Mapping[str, str]:
        request = flask.request
        with Timer() as timer:
            relayed_response = self.relay(request)
        if self.verbose:
            print('*****************')
            print('STORAGE FILE REQUEST:')
            print('********PATH*********')
            print(path)
            print('********ENDPATH*********')
            print(request.get_json())
            print('STORAGE FILE RESPONSE:')
            print(relayed_response.json())
            print('*****************')
        self.snoop_context(request, relayed_response, timer.elapsed, path=path)
        return relayed_response.json()

    def control(self) -> Mapping[str, str]:
        assert self.relay_control
        return self.relay_control.control(flask.request)
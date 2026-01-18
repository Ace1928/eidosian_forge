from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
def _start_message_handler(f):

    @components.Component.message_handler
    def handle(self, request):
        assert request.is_request('launch', 'attach')
        if self._initialize_request is None:
            raise request.isnt_valid('Session is not initialized yet')
        if self.launcher or self.server:
            raise request.isnt_valid('Session is already started')
        self.session.no_debug = request('noDebug', json.default(False))
        if self.session.no_debug:
            servers.dont_wait_for_first_connection()
        self.session.debug_options = debug_options = set(request('debugOptions', json.array(str)))
        f(self, request)
        if request.response is not None:
            return
        if self.server:
            self.server.initialize(self._initialize_request)
            self._initialize_request = None
            arguments = request.arguments
            if self.launcher:
                redirecting = arguments.get('console') == 'internalConsole'
                if 'RedirectOutput' in debug_options:
                    arguments = dict(arguments)
                    arguments['debugOptions'] = list(debug_options - {'RedirectOutput'})
                    redirecting = True
                if arguments.get('redirectOutput'):
                    arguments = dict(arguments)
                    del arguments['redirectOutput']
                    redirecting = True
                arguments['isOutputRedirected'] = redirecting
            try:
                self.server.channel.request(request.command, arguments)
            except messaging.NoMoreMessages:
                request.respond({})
                self.session.finalize('{0} disconnected before responding to {1}'.format(self.server, json.repr(request.command)))
                return
            except messaging.MessageHandlingError as exc:
                exc.propagate(request)
        if self.session.no_debug:
            self.start_request = request
            self.has_started = True
            request.respond({})
            self._propagate_deferred_events()
            return
        self.channel.send_event('initialized')
        self.start_request = request
        return messaging.NO_RESPONSE
    return handle
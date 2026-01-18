from __future__ import annotations
import logging
from pymongo import monitoring
class CommandLogger(monitoring.CommandListener):
    """A simple listener that logs command events.

    Listens for :class:`~pymongo.monitoring.CommandStartedEvent`,
    :class:`~pymongo.monitoring.CommandSucceededEvent` and
    :class:`~pymongo.monitoring.CommandFailedEvent` events and
    logs them at the `INFO` severity level using :mod:`logging`.
    .. versionadded:: 3.11
    """

    def started(self, event: monitoring.CommandStartedEvent) -> None:
        logging.info(f'Command {event.command_name} with request id {event.request_id} started on server {event.connection_id}')

    def succeeded(self, event: monitoring.CommandSucceededEvent) -> None:
        logging.info(f'Command {event.command_name} with request id {event.request_id} on server {event.connection_id} succeeded in {event.duration_micros} microseconds')

    def failed(self, event: monitoring.CommandFailedEvent) -> None:
        logging.info(f'Command {event.command_name} with request id {event.request_id} on server {event.connection_id} failed in {event.duration_micros} microseconds')
from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class _EventListeners:
    """Configure event listeners for a client instance.

    Any event listeners registered globally are included by default.

    :Parameters:
      - `listeners`: A list of event listeners.
    """

    def __init__(self, listeners: Optional[Sequence[_EventListener]]):
        self.__command_listeners = _LISTENERS.command_listeners[:]
        self.__server_listeners = _LISTENERS.server_listeners[:]
        lst = _LISTENERS.server_heartbeat_listeners
        self.__server_heartbeat_listeners = lst[:]
        self.__topology_listeners = _LISTENERS.topology_listeners[:]
        self.__cmap_listeners = _LISTENERS.cmap_listeners[:]
        if listeners is not None:
            for lst in listeners:
                if isinstance(lst, CommandListener):
                    self.__command_listeners.append(lst)
                if isinstance(lst, ServerListener):
                    self.__server_listeners.append(lst)
                if isinstance(lst, ServerHeartbeatListener):
                    self.__server_heartbeat_listeners.append(lst)
                if isinstance(lst, TopologyListener):
                    self.__topology_listeners.append(lst)
                if isinstance(lst, ConnectionPoolListener):
                    self.__cmap_listeners.append(lst)
        self.__enabled_for_commands = bool(self.__command_listeners)
        self.__enabled_for_server = bool(self.__server_listeners)
        self.__enabled_for_server_heartbeat = bool(self.__server_heartbeat_listeners)
        self.__enabled_for_topology = bool(self.__topology_listeners)
        self.__enabled_for_cmap = bool(self.__cmap_listeners)

    @property
    def enabled_for_commands(self) -> bool:
        """Are any CommandListener instances registered?"""
        return self.__enabled_for_commands

    @property
    def enabled_for_server(self) -> bool:
        """Are any ServerListener instances registered?"""
        return self.__enabled_for_server

    @property
    def enabled_for_server_heartbeat(self) -> bool:
        """Are any ServerHeartbeatListener instances registered?"""
        return self.__enabled_for_server_heartbeat

    @property
    def enabled_for_topology(self) -> bool:
        """Are any TopologyListener instances registered?"""
        return self.__enabled_for_topology

    @property
    def enabled_for_cmap(self) -> bool:
        """Are any ConnectionPoolListener instances registered?"""
        return self.__enabled_for_cmap

    def event_listeners(self) -> list[_EventListeners]:
        """List of registered event listeners."""
        return self.__command_listeners + self.__server_heartbeat_listeners + self.__server_listeners + self.__topology_listeners + self.__cmap_listeners

    def publish_command_start(self, command: _DocumentOut, database_name: str, request_id: int, connection_id: _Address, op_id: Optional[int]=None, service_id: Optional[ObjectId]=None) -> None:
        """Publish a CommandStartedEvent to all command listeners.

        :Parameters:
          - `command`: The command document.
          - `database_name`: The name of the database this command was run
            against.
          - `request_id`: The request id for this operation.
          - `connection_id`: The address (host, port) of the server this
            command was sent to.
          - `op_id`: The (optional) operation id for this operation.
          - `service_id`: The service_id this command was sent to, or ``None``.
        """
        if op_id is None:
            op_id = request_id
        event = CommandStartedEvent(command, database_name, request_id, connection_id, op_id, service_id=service_id)
        for subscriber in self.__command_listeners:
            try:
                subscriber.started(event)
            except Exception:
                _handle_exception()

    def publish_command_success(self, duration: timedelta, reply: _DocumentOut, command_name: str, request_id: int, connection_id: _Address, op_id: Optional[int]=None, service_id: Optional[ObjectId]=None, speculative_hello: bool=False, database_name: str='') -> None:
        """Publish a CommandSucceededEvent to all command listeners.

        :Parameters:
          - `duration`: The command duration as a datetime.timedelta.
          - `reply`: The server reply document.
          - `command_name`: The command name.
          - `request_id`: The request id for this operation.
          - `connection_id`: The address (host, port) of the server this
            command was sent to.
          - `op_id`: The (optional) operation id for this operation.
          - `service_id`: The service_id this command was sent to, or ``None``.
          - `speculative_hello`: Was the command sent with speculative auth?
          - `database_name`: The database this command was sent to, or ``""``.
        """
        if op_id is None:
            op_id = request_id
        if speculative_hello:
            reply = {}
        event = CommandSucceededEvent(duration, reply, command_name, request_id, connection_id, op_id, service_id, database_name=database_name)
        for subscriber in self.__command_listeners:
            try:
                subscriber.succeeded(event)
            except Exception:
                _handle_exception()

    def publish_command_failure(self, duration: timedelta, failure: _DocumentOut, command_name: str, request_id: int, connection_id: _Address, op_id: Optional[int]=None, service_id: Optional[ObjectId]=None, database_name: str='') -> None:
        """Publish a CommandFailedEvent to all command listeners.

        :Parameters:
          - `duration`: The command duration as a datetime.timedelta.
          - `failure`: The server reply document or failure description
            document.
          - `command_name`: The command name.
          - `request_id`: The request id for this operation.
          - `connection_id`: The address (host, port) of the server this
            command was sent to.
          - `op_id`: The (optional) operation id for this operation.
          - `service_id`: The service_id this command was sent to, or ``None``.
          - `database_name`: The database this command was sent to, or ``""``.
        """
        if op_id is None:
            op_id = request_id
        event = CommandFailedEvent(duration, failure, command_name, request_id, connection_id, op_id, service_id=service_id, database_name=database_name)
        for subscriber in self.__command_listeners:
            try:
                subscriber.failed(event)
            except Exception:
                _handle_exception()

    def publish_server_heartbeat_started(self, connection_id: _Address, awaited: bool) -> None:
        """Publish a ServerHeartbeatStartedEvent to all server heartbeat
        listeners.

        :Parameters:
         - `connection_id`: The address (host, port) pair of the connection.
         - `awaited`: True if this heartbeat is part of an awaitable hello command.
        """
        event = ServerHeartbeatStartedEvent(connection_id, awaited)
        for subscriber in self.__server_heartbeat_listeners:
            try:
                subscriber.started(event)
            except Exception:
                _handle_exception()

    def publish_server_heartbeat_succeeded(self, connection_id: _Address, duration: float, reply: Hello, awaited: bool) -> None:
        """Publish a ServerHeartbeatSucceededEvent to all server heartbeat
        listeners.

        :Parameters:
         - `connection_id`: The address (host, port) pair of the connection.
         - `duration`: The execution time of the event in the highest possible
            resolution for the platform.
         - `reply`: The command reply.
         - `awaited`: True if the response was awaited.
        """
        event = ServerHeartbeatSucceededEvent(duration, reply, connection_id, awaited)
        for subscriber in self.__server_heartbeat_listeners:
            try:
                subscriber.succeeded(event)
            except Exception:
                _handle_exception()

    def publish_server_heartbeat_failed(self, connection_id: _Address, duration: float, reply: Exception, awaited: bool) -> None:
        """Publish a ServerHeartbeatFailedEvent to all server heartbeat
        listeners.

        :Parameters:
         - `connection_id`: The address (host, port) pair of the connection.
         - `duration`: The execution time of the event in the highest possible
            resolution for the platform.
         - `reply`: The command reply.
         - `awaited`: True if the response was awaited.
        """
        event = ServerHeartbeatFailedEvent(duration, reply, connection_id, awaited)
        for subscriber in self.__server_heartbeat_listeners:
            try:
                subscriber.failed(event)
            except Exception:
                _handle_exception()

    def publish_server_opened(self, server_address: _Address, topology_id: ObjectId) -> None:
        """Publish a ServerOpeningEvent to all server listeners.

        :Parameters:
         - `server_address`: The address (host, port) pair of the server.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
        event = ServerOpeningEvent(server_address, topology_id)
        for subscriber in self.__server_listeners:
            try:
                subscriber.opened(event)
            except Exception:
                _handle_exception()

    def publish_server_closed(self, server_address: _Address, topology_id: ObjectId) -> None:
        """Publish a ServerClosedEvent to all server listeners.

        :Parameters:
         - `server_address`: The address (host, port) pair of the server.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
        event = ServerClosedEvent(server_address, topology_id)
        for subscriber in self.__server_listeners:
            try:
                subscriber.closed(event)
            except Exception:
                _handle_exception()

    def publish_server_description_changed(self, previous_description: ServerDescription, new_description: ServerDescription, server_address: _Address, topology_id: ObjectId) -> None:
        """Publish a ServerDescriptionChangedEvent to all server listeners.

        :Parameters:
         - `previous_description`: The previous server description.
         - `server_address`: The address (host, port) pair of the server.
         - `new_description`: The new server description.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
        event = ServerDescriptionChangedEvent(previous_description, new_description, server_address, topology_id)
        for subscriber in self.__server_listeners:
            try:
                subscriber.description_changed(event)
            except Exception:
                _handle_exception()

    def publish_topology_opened(self, topology_id: ObjectId) -> None:
        """Publish a TopologyOpenedEvent to all topology listeners.

        :Parameters:
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
        event = TopologyOpenedEvent(topology_id)
        for subscriber in self.__topology_listeners:
            try:
                subscriber.opened(event)
            except Exception:
                _handle_exception()

    def publish_topology_closed(self, topology_id: ObjectId) -> None:
        """Publish a TopologyClosedEvent to all topology listeners.

        :Parameters:
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
        event = TopologyClosedEvent(topology_id)
        for subscriber in self.__topology_listeners:
            try:
                subscriber.closed(event)
            except Exception:
                _handle_exception()

    def publish_topology_description_changed(self, previous_description: TopologyDescription, new_description: TopologyDescription, topology_id: ObjectId) -> None:
        """Publish a TopologyDescriptionChangedEvent to all topology listeners.

        :Parameters:
         - `previous_description`: The previous topology description.
         - `new_description`: The new topology description.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
        event = TopologyDescriptionChangedEvent(previous_description, new_description, topology_id)
        for subscriber in self.__topology_listeners:
            try:
                subscriber.description_changed(event)
            except Exception:
                _handle_exception()

    def publish_pool_created(self, address: _Address, options: dict[str, Any]) -> None:
        """Publish a :class:`PoolCreatedEvent` to all pool listeners."""
        event = PoolCreatedEvent(address, options)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.pool_created(event)
            except Exception:
                _handle_exception()

    def publish_pool_ready(self, address: _Address) -> None:
        """Publish a :class:`PoolReadyEvent` to all pool listeners."""
        event = PoolReadyEvent(address)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.pool_ready(event)
            except Exception:
                _handle_exception()

    def publish_pool_cleared(self, address: _Address, service_id: Optional[ObjectId]) -> None:
        """Publish a :class:`PoolClearedEvent` to all pool listeners."""
        event = PoolClearedEvent(address, service_id)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.pool_cleared(event)
            except Exception:
                _handle_exception()

    def publish_pool_closed(self, address: _Address) -> None:
        """Publish a :class:`PoolClosedEvent` to all pool listeners."""
        event = PoolClosedEvent(address)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.pool_closed(event)
            except Exception:
                _handle_exception()

    def publish_connection_created(self, address: _Address, connection_id: int) -> None:
        """Publish a :class:`ConnectionCreatedEvent` to all connection
        listeners.
        """
        event = ConnectionCreatedEvent(address, connection_id)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_created(event)
            except Exception:
                _handle_exception()

    def publish_connection_ready(self, address: _Address, connection_id: int) -> None:
        """Publish a :class:`ConnectionReadyEvent` to all connection listeners."""
        event = ConnectionReadyEvent(address, connection_id)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_ready(event)
            except Exception:
                _handle_exception()

    def publish_connection_closed(self, address: _Address, connection_id: int, reason: str) -> None:
        """Publish a :class:`ConnectionClosedEvent` to all connection
        listeners.
        """
        event = ConnectionClosedEvent(address, connection_id, reason)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_closed(event)
            except Exception:
                _handle_exception()

    def publish_connection_check_out_started(self, address: _Address) -> None:
        """Publish a :class:`ConnectionCheckOutStartedEvent` to all connection
        listeners.
        """
        event = ConnectionCheckOutStartedEvent(address)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_check_out_started(event)
            except Exception:
                _handle_exception()

    def publish_connection_check_out_failed(self, address: _Address, reason: str) -> None:
        """Publish a :class:`ConnectionCheckOutFailedEvent` to all connection
        listeners.
        """
        event = ConnectionCheckOutFailedEvent(address, reason)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_check_out_failed(event)
            except Exception:
                _handle_exception()

    def publish_connection_checked_out(self, address: _Address, connection_id: int) -> None:
        """Publish a :class:`ConnectionCheckedOutEvent` to all connection
        listeners.
        """
        event = ConnectionCheckedOutEvent(address, connection_id)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_checked_out(event)
            except Exception:
                _handle_exception()

    def publish_connection_checked_in(self, address: _Address, connection_id: int) -> None:
        """Publish a :class:`ConnectionCheckedInEvent` to all connection
        listeners.
        """
        event = ConnectionCheckedInEvent(address, connection_id)
        for subscriber in self.__cmap_listeners:
            try:
                subscriber.connection_checked_in(event)
            except Exception:
                _handle_exception()
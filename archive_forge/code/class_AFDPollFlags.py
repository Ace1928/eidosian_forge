from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class AFDPollFlags(enum.IntFlag):
    AFD_POLL_RECEIVE = 1
    AFD_POLL_RECEIVE_EXPEDITED = 2
    AFD_POLL_SEND = 4
    AFD_POLL_DISCONNECT = 8
    AFD_POLL_ABORT = 16
    AFD_POLL_LOCAL_CLOSE = 32
    AFD_POLL_CONNECT = 64
    AFD_POLL_ACCEPT = 128
    AFD_POLL_CONNECT_FAIL = 256
    AFD_POLL_QOS = 512
    AFD_POLL_GROUP_QOS = 1024
    AFD_POLL_ROUTING_INTERFACE_CHANGE = 2048
    AFD_POLL_EVENT_ADDRESS_LIST_CHANGE = 4096
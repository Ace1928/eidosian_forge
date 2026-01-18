from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class SocketOption(IntEnum):
    """Options for Socket.get/set

    .. versionadded:: 23
    """
    _opt_type: _OptType

    def __new__(cls, value: int, opt_type: _OptType=_OptType.int):
        """Attach option type as `._opt_type`"""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._opt_type = opt_type
        return obj
    HWM = 1
    AFFINITY = (4, _OptType.int64)
    ROUTING_ID = (5, _OptType.bytes)
    SUBSCRIBE = (6, _OptType.bytes)
    UNSUBSCRIBE = (7, _OptType.bytes)
    RATE = 8
    RECOVERY_IVL = 9
    SNDBUF = 11
    RCVBUF = 12
    RCVMORE = 13
    FD = (14, _OptType.fd)
    EVENTS = 15
    TYPE = 16
    LINGER = 17
    RECONNECT_IVL = 18
    BACKLOG = 19
    RECONNECT_IVL_MAX = 21
    MAXMSGSIZE = (22, _OptType.int64)
    SNDHWM = 23
    RCVHWM = 24
    MULTICAST_HOPS = 25
    RCVTIMEO = 27
    SNDTIMEO = 28
    LAST_ENDPOINT = (32, _OptType.bytes)
    ROUTER_MANDATORY = 33
    TCP_KEEPALIVE = 34
    TCP_KEEPALIVE_CNT = 35
    TCP_KEEPALIVE_IDLE = 36
    TCP_KEEPALIVE_INTVL = 37
    IMMEDIATE = 39
    XPUB_VERBOSE = 40
    ROUTER_RAW = 41
    IPV6 = 42
    MECHANISM = 43
    PLAIN_SERVER = 44
    PLAIN_USERNAME = (45, _OptType.bytes)
    PLAIN_PASSWORD = (46, _OptType.bytes)
    CURVE_SERVER = 47
    CURVE_PUBLICKEY = (48, _OptType.bytes)
    CURVE_SECRETKEY = (49, _OptType.bytes)
    CURVE_SERVERKEY = (50, _OptType.bytes)
    PROBE_ROUTER = 51
    REQ_CORRELATE = 52
    REQ_RELAXED = 53
    CONFLATE = 54
    ZAP_DOMAIN = (55, _OptType.bytes)
    ROUTER_HANDOVER = 56
    TOS = 57
    CONNECT_ROUTING_ID = (61, _OptType.bytes)
    GSSAPI_SERVER = 62
    GSSAPI_PRINCIPAL = (63, _OptType.bytes)
    GSSAPI_SERVICE_PRINCIPAL = (64, _OptType.bytes)
    GSSAPI_PLAINTEXT = 65
    HANDSHAKE_IVL = 66
    SOCKS_PROXY = (68, _OptType.bytes)
    XPUB_NODROP = 69
    BLOCKY = 70
    XPUB_MANUAL = 71
    XPUB_WELCOME_MSG = (72, _OptType.bytes)
    STREAM_NOTIFY = 73
    INVERT_MATCHING = 74
    HEARTBEAT_IVL = 75
    HEARTBEAT_TTL = 76
    HEARTBEAT_TIMEOUT = 77
    XPUB_VERBOSER = 78
    CONNECT_TIMEOUT = 79
    TCP_MAXRT = 80
    THREAD_SAFE = 81
    MULTICAST_MAXTPDU = 84
    VMCI_BUFFER_SIZE = (85, _OptType.int64)
    VMCI_BUFFER_MIN_SIZE = (86, _OptType.int64)
    VMCI_BUFFER_MAX_SIZE = (87, _OptType.int64)
    VMCI_CONNECT_TIMEOUT = 88
    USE_FD = 89
    GSSAPI_PRINCIPAL_NAMETYPE = 90
    GSSAPI_SERVICE_PRINCIPAL_NAMETYPE = 91
    BINDTODEVICE = (92, _OptType.bytes)
    IDENTITY = ROUTING_ID
    CONNECT_RID = CONNECT_ROUTING_ID
    TCP_ACCEPT_FILTER = (38, _OptType.bytes)
    IPC_FILTER_PID = 58
    IPC_FILTER_UID = 59
    IPC_FILTER_GID = 60
    IPV4ONLY = 31
    DELAY_ATTACH_ON_CONNECT = IMMEDIATE
    FAIL_UNROUTABLE = ROUTER_MANDATORY
    ROUTER_BEHAVIOR = ROUTER_MANDATORY
    ZAP_ENFORCE_DOMAIN = 93
    LOOPBACK_FASTPATH = 94
    METADATA = (95, _OptType.bytes)
    MULTICAST_LOOP = 96
    ROUTER_NOTIFY = 97
    XPUB_MANUAL_LAST_VALUE = 98
    SOCKS_USERNAME = (99, _OptType.bytes)
    SOCKS_PASSWORD = (100, _OptType.bytes)
    IN_BATCH_SIZE = 101
    OUT_BATCH_SIZE = 102
    WSS_KEY_PEM = (103, _OptType.bytes)
    WSS_CERT_PEM = (104, _OptType.bytes)
    WSS_TRUST_PEM = (105, _OptType.bytes)
    WSS_HOSTNAME = (106, _OptType.bytes)
    WSS_TRUST_SYSTEM = 107
    ONLY_FIRST_SUBSCRIBE = 108
    RECONNECT_STOP = 109
    HELLO_MSG = (110, _OptType.bytes)
    DISCONNECT_MSG = (111, _OptType.bytes)
    PRIORITY = 112
    BUSY_POLL = 113
    HICCUP_MSG = (114, _OptType.bytes)
    XSUB_VERBOSE_UNSUBSCRIBE = 115
    TOPICS_COUNT = 116
    NORM_MODE = 117
    NORM_UNICAST_NACK = 118
    NORM_BUFFER_SIZE = 119
    NORM_SEGMENT_SIZE = 120
    NORM_BLOCK_SIZE = 121
    NORM_NUM_PARITY = 122
    NORM_NUM_AUTOPARITY = 123
    NORM_PUSH = 124